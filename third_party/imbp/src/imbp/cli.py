"""Командная строка для пакетной обработки снимков.

Примеры:
    uv run imbp analyze "../ML/raw_images/E. coli, 1.bmp"                 # «Авто», результат в ./results
    uv run imbp analyze ../ML/raw_images --method gmm --out results_gmm   # вся папка, метод GMM
    uv run imbp analyze снимок.bmp --method all                           # все методы (auto, slic, gmm)
    uv run imbp info                                                      # сведения о модели

Для каждого снимка сохраняются:
    <имя>_<метод>.png   — снимок с границами сегментации и эллипсами (зел. — кокки, син. — бациллы);
    <имя>_<метод>_mask.png — бинарная маска бактерий;
    <имя>_<метод>.json  — сводка и параметры каждого объекта.
Плюс общий summary.csv по всем снимкам.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import config
from .io import IMAGE_EXTENSIONS, list_images, load_image, save_image
from .params_model import SlicParamPredictor
from .pipeline import METHODS, Analyzer


def _collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(list_images(p))
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
        else:
            print(f"Пропуск (не изображение): {p}")
    return files


def cmd_analyze(args: argparse.Namespace) -> int:
    files = _collect_inputs(args.inputs)
    if not files:
        print("Нет изображений для обработки.")
        return 1
    methods = list(METHODS) if args.method == "all" else [args.method]
    analyzer = Analyzer(model_path=args.model, use_filter=not args.no_filter)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        try:
            img = load_image(path)
        except ValueError as exc:
            print(f"Ошибка: {exc}")
            continue
        for method in methods:
            t0 = time.perf_counter()
            res = analyzer.analyze(img, method)
            elapsed = time.perf_counter() - t0
            stem = f"{path.stem}_{method}"
            save_image(args.out / f"{stem}.png", res.overlay)
            save_image(args.out / f"{stem}_mask.png", res.mask)
            info = {"file": str(path), "seconds": round(elapsed, 3), **res.to_dict(include_objects=not args.no_objects)}
            (args.out / f"{stem}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({"file": path.name, "method": method, "seconds": round(elapsed, 3),
                         "bacteria_area_fraction": info["bacteria_area_fraction"], **res.summary.to_dict()})
            print(f"{path.name} [{method}, {elapsed:.1f} с]: {res.summary.text().replace(chr(10), ' | ')}")

    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nРезультаты: {args.out.resolve()}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print(f"Данные (IMBP_DATA_DIR): {config.DATA_DIR}")
    print(f"Модель (IMBP_MODEL_PATH): {args.model}")
    try:
        predictor = SlicParamPredictor.load(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  модель недоступна: {exc}")
        return 1
    print(json.dumps(predictor.meta, ensure_ascii=False, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imbp", description="Сегментация и классификация бактерий на микроснимках")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH, help="путь к модели RF (.joblib)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="обработать снимки (файлы или папки)")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--method", choices=[*METHODS, "all"], default="auto",
                   help="метод сегментации: auto — выбор по тайлам (по умолчанию), slic, gmm, all — все")
    p.add_argument("--no-filter", action="store_true", help="не применять фильтр ложных объектов")
    p.add_argument("--out", type=Path, default=Path("results"), help="папка для результатов")
    p.add_argument("--no-objects", action="store_true", help="не сохранять список объектов в JSON")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("info", help="показать сведения о модели и путях")
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8 — без этого падает вывод кириллицы.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
