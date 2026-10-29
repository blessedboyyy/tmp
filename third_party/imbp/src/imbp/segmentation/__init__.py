"""Методы сегментации: SLIC с параметрами от Random Forest и GMM с пространственной очисткой.

Оба метода возвращают объект с полями ``mask`` (bool, бактерии) и ``labels`` (int32, 0 — фон),
поэтому дальше (морфология, визуализация) с ними можно работать одинаково.
"""

from .fusion import AutoResult, MethodChooser, segment_auto
from .gmm import GmmResult, gmm_clusters, segment_gmm, select_bacteria_clusters
from .slic import SlicResult, TileInfo, oracle_mask, run_slic, segment_slic, tile_grid, vote_superpixels

__all__ = [
    "AutoResult",
    "MethodChooser",
    "segment_auto",
    "GmmResult",
    "SlicResult",
    "TileInfo",
    "gmm_clusters",
    "oracle_mask",
    "run_slic",
    "segment_gmm",
    "segment_slic",
    "select_bacteria_clusters",
    "tile_grid",
    "vote_superpixels",
]
