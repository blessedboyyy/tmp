"""Сегментация моделью гауссовых смесей (GMM) с приближённой пространственной регуляризацией (раздел 4.2).

Конвейер:
    1. двусторонний (bilateral) фильтр — подавляет шум, сохраняя границы;
    2. GMM из 3 компонент (фон / микроорганизмы / артефакты) по цвету пикселя, EM-алгоритм;
    3. HMRF (раздел 4.2 курсовой, формула энергии): MAP-оценка меток методом ICM —
       E(x) = Σ −log p(x_i | y_i) + β Σ_{j∈N(i)} [x_i ≠ x_j] (потенциал Поттса, 8 соседей).
       В курсовой HMRF только аппроксимировался фильтрацией; здесь он реализован (``config.GMM_HMRF_BETA``;
       β = 0 — выключено, остаётся аппроксимация как в курсовой);
    4. удаление связных компонент площадью < ``GMM_MIN_AREA`` (убирает «соль/перец»).

Метод устойчивее SLIC к расфокусу и градиентным пятнам красителя, но может терять локальную
геометрию — поэтому в ПО пользователь выбирает метод сам (мультимодальный подход).

Отличия от исходного кода (step8/step12/step13):
    * EM обучается на случайной подвыборке пикселей (``GMM_FIT_SAMPLE``), предсказание — для всех:
      на снимке 1616×1216 это в ~10 раз быстрее при практически тех же центрах кластеров;
    * перенумерация кластеров через lookup-массив вместо медленного ``np.vectorize``;
    * «бактериальные» кластеры выбираются адаптивно (``select_bacteria_clusters``), а не жёстко
      средний кластер, как в исходном коде (там MIoU был 15.8%, сейчас 47.5% на тесте).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .. import config
from ..masks import otsu_bacteria_mask, remove_small_components


@dataclass
class GmmResult:
    """Результат GMM-сегментации.

    Attributes:
        mask: bool (H, W) — пиксели бактерий (после очистки).
        labels: int32 (H, W) — связные компоненты маски (0 — фон); ``Analyzer.segment`` заменяет их
            картой отдельных клеток (instances.split_instances).
        clusters: uint8 (H, W) — номер кластера каждого пикселя после сортировки по яркости
            (0 — самый тёмный, ``n_components-1`` — самый светлый), до очистки.
        cluster_means: (n_components, 3) — центры кластеров в RGB, в том же порядке.
        is_empty: True, если по критерию Оцу бактерий нет (и включена проверка ``empty_check``).
        bacteria_clusters: какие кластеры были приняты за бактерии.
    """

    mask: np.ndarray
    labels: np.ndarray
    clusters: np.ndarray
    cluster_means: np.ndarray
    is_empty: bool = False
    bacteria_clusters: tuple[int, ...] = config.GMM_BACTERIA_CLUSTERS
    n_rejected: int = 0  # сколько компонент удалил фильтр ложных объектов (заполняет pipeline)

    def describe(self) -> str:
        if self.is_empty:
            return "GMM: бактерии не обнаружены"
        suffix = f"; фильтр убрал {self.n_rejected} объект(ов)" if self.n_rejected else ""
        return f"GMM ({self.cluster_means.shape[0]} кластера, бактерии — {list(self.bacteria_clusters)}){suffix}"


def hmrf_icm(log_post: np.ndarray, beta: float, iterations: int) -> np.ndarray:
    """MAP-разметка скрытого марковского поля методом ICM (Besag, 1986).

    Args:
        log_post: (H, W, K) — логарифмы апостериорных вероятностей классов из GMM (унарный член).
        beta: вес потенциала Поттса — штраф за каждого из 8 соседей с другой меткой.
        iterations: число проходов ICM.
    Returns:
        Карта меток int (H, W).

    Обновление «шахматное»: сначала пиксели одного цвета клетки, потом другого. Соседи пикселя
    всегда другого цвета, поэтому каждое полу-обновление — точный покоординатный спуск по энергии
    (в отличие от одновременного обновления, которое может зацикливаться).
    """
    h, w, k = log_post.shape
    labels = log_post.argmax(axis=2)
    if beta <= 0 or iterations <= 0:
        return labels
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    checker = (np.add.outer(np.arange(h), np.arange(w)) % 2).astype(bool)
    for _ in range(iterations):
        changed = 0
        for phase in (checker, ~checker):
            # Число соседей с меткой c для каждого класса c: свёртка индикатора метки.
            counts = np.stack([cv2.filter2D((labels == c).astype(np.float32), -1, kernel,
                                            borderType=cv2.BORDER_REPLICATE) for c in range(k)], axis=2)
            # Минимум энергии = максимум log p_c + β·(число соседей класса c).
            new = (log_post + beta * counts).argmax(axis=2)
            upd = phase & (new != labels)
            changed += int(upd.sum())
            labels = np.where(phase, new, labels)
        if changed == 0:
            break
    return labels


def gmm_clusters(
    img_bgr: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    fit_sample: int | None = config.GMM_FIT_SAMPLE,
    random_state: int = config.RANDOM_STATE,
    hmrf_beta: float = config.GMM_HMRF_BETA,
    hmrf_iterations: int = config.GMM_HMRF_ITERS,
    bilateral: bool = config.GMM_BILATERAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Кластеризация пикселей: (карта кластеров uint8 (H, W), центры (n, 3) в RGB).

    Кластеры упорядочены по средней яркости центра: 0 — самый тёмный.
    При ``hmrf_beta`` > 0 метки уточняются HMRF (ICM), иначе — независимо по пикселям (как в курсовой).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    filtered = cv2.bilateralFilter(img_rgb, d=config.GMM_BILATERAL_D,
                                   sigmaColor=config.GMM_BILATERAL_SIGMA_COLOR,
                                   sigmaSpace=config.GMM_BILATERAL_SIGMA_SPACE) if bilateral else img_rgb
    pixels = filtered.reshape(-1, 3).astype(np.float64)

    rng = np.random.default_rng(random_state)
    if fit_sample and len(pixels) > fit_sample:
        train = pixels[rng.choice(len(pixels), fit_sample, replace=False)]
    else:
        train = pixels

    gmm = GaussianMixture(n_components=n_components, covariance_type=config.GMM_COVARIANCE_TYPE,
                          random_state=random_state)
    with warnings.catch_warnings():
        # На почти однотонных изображениях различных цветов меньше, чем кластеров, — sklearn
        # предупреждает об этом. Результат при этом корректен (лишние кластеры пустые), глушим.
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm.fit(train)
    if hmrf_beta > 0:
        log_post = np.log(gmm.predict_proba(pixels) + 1e-12).reshape(*img_bgr.shape[:2], n_components)
        raw = hmrf_icm(log_post, hmrf_beta, hmrf_iterations)
    else:
        raw = gmm.predict(pixels)

    # Сортировка кластеров по яркости: lookup[старый_номер] = новый_номер.
    order = np.argsort(gmm.means_.mean(axis=1))
    lookup = np.empty(n_components, dtype=np.uint8)
    lookup[order] = np.arange(n_components, dtype=np.uint8)
    clusters = lookup[raw].reshape(img_bgr.shape[:2])
    return clusters, gmm.means_[order]


def select_bacteria_clusters(
    clusters: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    base: tuple[int, ...] = config.GMM_BACTERIA_CLUSTERS,
    extra_max_fraction: float = config.GMM_EXTRA_CLUSTER_MAX_FRACTION,
) -> tuple[int, ...]:
    """Номера кластеров-бактерий: ``base`` + небольшие по площади кластеры, кроме самого светлого (фон).

    Args:
        clusters: карта кластеров после сортировки по яркости (результат ``gmm_clusters``).
        base: кластеры, которые всегда считаются бактериями.
        extra_max_fraction: максимальная доля площади, при которой прочий кластер тоже относится к бактериям
            (0 — отключить адаптивное добавление).
    """
    fractions = np.bincount(clusters.ravel(), minlength=n_components) / clusters.size
    extra = [c for c in range(n_components - 1) if c not in base and fractions[c] < extra_max_fraction]
    return tuple(sorted({*base, *extra}))


def segment_gmm(
    img_bgr: np.ndarray,
    bacteria_clusters: tuple[int, ...] | None = None,
    min_area: int = config.GMM_MIN_AREA,
    empty_check: bool = True,
    otsu_threshold: float | None = None,
    **cluster_kwargs,
) -> GmmResult:
    """Полный конвейер GMM-сегментации.

    Args:
        img_bgr: снимок BGR uint8.
        bacteria_clusters: номера кластеров (после сортировки по яркости), считающиеся бактериями.
            None — адаптивный выбор ``select_bacteria_clusters`` (по умолчанию).
        min_area: минимальная площадь компоненты, меньшие удаляются.
        empty_check: если True и по критерию Оцу снимок однородный — вернуть пустую маску
            (GMM всегда находит 3 кластера, даже на чистом фоне, и размечает шум как бактерии).
        otsu_threshold: порог Оцу для проверки на пустоту (None — по этому изображению).
        cluster_kwargs: доп. параметры для ``gmm_clusters`` (n_components, fit_sample, random_state).
    """
    h, w = img_bgr.shape[:2]
    if empty_check and otsu_bacteria_mask(img_bgr, threshold=otsu_threshold).is_empty:
        n = cluster_kwargs.get("n_components", config.GMM_N_COMPONENTS)
        return GmmResult(np.zeros((h, w), bool), np.zeros((h, w), np.int32),
                         np.full((h, w), n - 1, np.uint8), np.zeros((n, 3)), is_empty=True,
                         bacteria_clusters=())

    clusters, means = gmm_clusters(img_bgr, **cluster_kwargs)
    if bacteria_clusters is None:
        bacteria_clusters = select_bacteria_clusters(clusters, len(means))
    mask = np.isin(clusters, bacteria_clusters)
    if min_area > 0:
        mask = remove_small_components(mask, min_area)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return GmmResult(mask=mask, labels=labels.astype(np.int32), clusters=clusters, cluster_means=means,
                     bacteria_clusters=tuple(bacteria_clusters))
