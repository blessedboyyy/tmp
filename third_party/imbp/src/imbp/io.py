"""Чтение и запись изображений с поддержкой кириллицы в путях.

``cv2.imread``/``cv2.imwrite`` на Windows не умеют открывать пути с не-ASCII символами
(а имена исходных снимков ИМБП содержат кириллицу: «1 Зев (1) 10 в 6, Грам + палочки...»).
Поэтому файл читается как массив байтов и декодируется через ``cv2.imdecode``.

Соглашение пакета: изображения — ``numpy.ndarray`` формы (H, W, 3), dtype ``uint8``, порядок каналов BGR
(как в OpenCV). Маски — ``bool`` (H, W).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Расширения, которые принимает ПО (снимки микроскопа приходят в .bmp и .jpg, патчи — .png).
IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_image(path: str | Path) -> np.ndarray:
    """Загрузить цветное изображение (BGR, uint8). Бросает ``ValueError``, если файл не декодируется."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось декодировать изображение: {path}")
    return img


def load_mask(path: str | Path) -> np.ndarray:
    """Загрузить бинарную маску (белое = объект) как ``bool``-массив."""
    data = np.fromfile(str(path), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Не удалось декодировать маску: {path}")
    return mask > 127


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Сохранить изображение (BGR или маску) в формат по расширению файла; создаёт папки при необходимости."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Не удалось закодировать изображение для {path}")
    buf.tofile(str(path))


def list_images(folder: str | Path) -> list[Path]:
    """Отсортированный список изображений в папке (без рекурсии)."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
