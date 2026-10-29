"""Работа с обучающими данными (нужно только для scripts/, в ПО не используется).

Структура данных (папка ``config.DATA_DIR``, по умолчанию ``../ML``):
    train_subset_images/img_XXXX.png        — патчи ≈404×405 (случайная подвыборка из dataset_patches)
    train_subset_masks/img_XXXX_mask.png           — эталон Cellpose cyto2 (псевдоразметка)
    train_subset_masks_omnipose/img_XXXX_mask.png  — эталон Omnipose bact_phase_omni
Какой эталон используется — ``config.REFERENCE`` (переменная окружения IMBP_REFERENCE).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .io import load_mask


def mask_path_for(image_path: str | Path, reference: str | None = None) -> Path:
    """Путь к эталонной маске патча: img_0017.png -> <папка эталона>/img_0017_mask.png.

    ``reference`` — "cellpose" / "omnipose"; None — текущий ``config.REFERENCE``.
    """
    masks_dir = config.REFERENCE_MASK_DIRS[reference or config.REFERENCE]
    return masks_dir / f"{Path(image_path).stem}_mask.png"


def subset_images(images_dir: Path = config.SUBSET_IMAGES_DIR) -> list[Path]:
    """Отсортированный список патчей обучающей подвыборки, для которых есть маска."""
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Нет папки с обучающими патчами {images_dir}. Укажите путь к данным через IMBP_DATA_DIR."
        )
    return [p for p in sorted(images_dir.glob("*.png")) if mask_path_for(p).exists()]


@lru_cache(maxsize=1)
def coursework_test_files() -> frozenset[str]:
    """Имена 80 тестовых патчей курсовой (не зависят от выбранного эталона).

    Курсовая: непустые по Cellpose патчи, отсортированные по имени, train_test_split(0.2, random_state=42).
    Тест фиксирован, чтобы метрики для разных эталонов и разных версий кода были сравнимы.
    """
    names = [p.name for p in sorted(config.SUBSET_IMAGES_DIR.glob("*.png"))
             if load_mask(mask_path_for(p, "cellpose")).any()]
    _, test = train_test_split(names, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    return frozenset(test)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разбиение таблицы со столбцом ``filename`` на train и фиксированный тест курсовой.

    Для эталона Cellpose это в точности разбиение курсовой (step5–step7, step13); для других эталонов
    тест — те же 80 патчей, а train — все остальные строки ``df``.
    """
    df = df.sort_values("filename").reset_index(drop=True)
    is_test = df["filename"].isin(coursework_test_files())
    return df[~is_test], df[is_test]
