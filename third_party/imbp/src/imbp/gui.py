"""Графический интерфейс на Tkinter (раздел 4.4 курсовой) — демонстрационная оболочка над ``pipeline``.

Сценарий как в ML/step12_final_app.py:
    1. загрузить снимок;
    2. выбрать метод сегментации («Авто», SLIC или GMM) — справа показываются границы сегментации;
    3. «Анализ морфологии и окраски» — эллипсы (зел. — кокки, син. — бациллы) и статистика.

Отличия: вся логика вынесена в ``imbp.pipeline.Analyzer`` (GUI ничего не считает сам), модель RF
загружается из файла один раз, тяжёлые вычисления выполняются в фоновом потоке — окно не «зависает».
Для интегратора: при переносе в своё ПО этот модуль можно не использовать.

Запуск:  uv run imbp-gui
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from .io import IMAGE_EXTENSIONS, load_image, save_image
from .pipeline import Analyzer
from .visualization import draw_boundaries, render_analysis

PREVIEW_SIZE = 480  # максимальная сторона превью в окне, пиксели


class App(tk.Tk):
    def __init__(self, analyzer: Analyzer):
        super().__init__()
        self.analyzer = analyzer
        self.title("IMBP: анализатор микробиоты")
        self.geometry("1150x780")
        self.img_bgr: np.ndarray | None = None
        self.segmentation = None      # SlicResult | GmmResult | AutoResult для текущего снимка
        self.result_bgr: np.ndarray | None = None
        self._build_ui()

    # ------------------------------------------------------------------ интерфейс
    def _build_ui(self) -> None:
        top = tk.Frame(self, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Button(top, text="1. Загрузить изображение", command=self.load_image,
                  font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=20)
        self.btn_save = tk.Button(top, text="Сохранить результат", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(side=tk.RIGHT, padx=20)

        mid = tk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20)
        self.panel_left = tk.Label(mid, text="Оригинал", bg="gray")
        self.panel_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.panel_right = tk.Label(mid, text="Результат", bg="gray")
        self.panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.lbl_status = tk.Label(self, text="Ожидание изображения...", font=("Arial", 13), justify=tk.LEFT)
        self.lbl_status.pack(pady=8)

        seg = tk.Frame(self, pady=6)
        seg.pack(side=tk.TOP)
        tk.Label(seg, text="2. Метод сегментации:", font=("Arial", 11)).pack(side=tk.LEFT, padx=10)
        self.btn_auto = tk.Button(seg, text="Авто (рекомендуется)", width=22, bg="khaki",
                                  command=lambda: self.run_segmentation("auto"), state=tk.DISABLED)
        self.btn_auto.pack(side=tk.LEFT, padx=10)
        self.btn_slic = tk.Button(seg, text="SLIC (Random Forest)", width=22, bg="lightblue",
                                  command=lambda: self.run_segmentation("slic"), state=tk.DISABLED)
        self.btn_slic.pack(side=tk.LEFT, padx=10)
        self.btn_gmm = tk.Button(seg, text="GMM", width=22, bg="lightgreen",
                                 command=lambda: self.run_segmentation("gmm"), state=tk.DISABLED)
        self.btn_gmm.pack(side=tk.LEFT, padx=10)
        # Фильтр ложных объектов можно отключить: он отбрасывает бледные/размытые объекты, а будут ли
        # они считаться клетками — решение лаборатории (см. README, «Ограничения»).
        self._loaded_filter = self.analyzer.object_filter
        self.var_filter = tk.BooleanVar(value=self._loaded_filter is not None)
        tk.Checkbutton(seg, text="Фильтр артефактов", variable=self.var_filter, command=self._toggle_filter,
                       state=tk.NORMAL if self._loaded_filter is not None else tk.DISABLED).pack(side=tk.LEFT, padx=10)

        bottom = tk.Frame(self, pady=12)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.btn_analyze = tk.Button(bottom, text="3. Анализ морфологии и окраски", width=40, height=2,
                                     bg="pink", font=("Arial", 12, "bold"), command=self.run_morphology,
                                     state=tk.DISABLED)
        self.btn_analyze.pack()

    def _toggle_filter(self) -> None:
        """Включить/выключить фильтр ложных объектов для следующих сегментаций."""
        self.analyzer.object_filter = self._loaded_filter if self.var_filter.get() else None

    def _set_status(self, text: str, color: str = "black") -> None:
        self.lbl_status.config(text=text, fg=color)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in (self.btn_auto, self.btn_slic, self.btn_gmm):
            btn.config(state=state if self.img_bgr is not None else tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED if busy or self.segmentation is None else tk.NORMAL)
        self.config(cursor="watch" if busy else "")

    def _show(self, img_bgr: np.ndarray, panel: tk.Label) -> None:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ratio = min(PREVIEW_SIZE / w, PREVIEW_SIZE / h)
        resized = cv2.resize(rgb, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        panel.config(image=photo, text="")
        panel.image = photo  # держим ссылку, иначе картинку удалит сборщик мусора

    def _run_in_background(self, work, on_done) -> None:
        """Выполнить ``work()`` в потоке, затем ``on_done(result)`` в потоке Tk."""
        self._set_busy(True)

        def target():
            try:
                res = work()
                self.after(0, lambda: (self._set_busy(False), on_done(res)))
            except Exception as exc:  # noqa: BLE001 — показать любую ошибку пользователю
                # exc передаём через аргумент по умолчанию: после except имя exc удаляется.
                self.after(0, lambda e=exc: (self._set_busy(False), messagebox.showerror("Ошибка", str(e))))

        threading.Thread(target=target, daemon=True).start()

    # ------------------------------------------------------------------ действия
    def load_image(self) -> None:
        types = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
        path = filedialog.askopenfilename(title="Выберите снимок",
                                          filetypes=(("Изображения", types), ("Все файлы", "*.*")))
        if not path:
            return
        try:
            self.img_bgr = load_image(path)
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        self.segmentation = None
        self.result_bgr = None
        self._show(self.img_bgr, self.panel_left)
        self.panel_right.config(image="", text="Результат")
        self.btn_save.config(state=tk.DISABLED)
        self._set_busy(False)
        h, w = self.img_bgr.shape[:2]
        self._set_status(f"Загружено {w}×{h}. Выберите метод сегментации (шаг 2).", "blue")

    def run_segmentation(self, method: str) -> None:
        if self.img_bgr is None:
            return
        img = self.img_bgr
        self._set_status(f"Сегментация ({method.upper()})...")

        def done(seg):
            self.segmentation = seg
            self.result_bgr = draw_boundaries(img, seg.labels)
            self._show(self.result_bgr, self.panel_right)
            self.btn_save.config(state=tk.NORMAL)
            self.btn_analyze.config(state=tk.NORMAL)
            self._set_status(f"{seg.describe()}\nНажмите «Анализ морфологии и окраски» (шаг 3).", "red")

        self._run_in_background(lambda: self.analyzer.segment(img, method), done)

    def run_morphology(self) -> None:
        if self.img_bgr is None or self.segmentation is None:
            return
        img, seg = self.img_bgr, self.segmentation
        self._set_status("Расчёт морфологических признаков...")

        def done(morph):
            self.result_bgr = render_analysis(img, seg.labels, morph)
            self._show(self.result_bgr, self.panel_right)
            self._set_status(morph.summary.text(), "green")

        self._run_in_background(lambda: self.analyzer.classify(img, seg), done)

    def save_result(self) -> None:
        if self.result_bgr is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG", "*.png"),))
        if path:
            save_image(path, self.result_bgr)


def main() -> None:
    App(Analyzer()).mainloop()


if __name__ == "__main__":
    main()
