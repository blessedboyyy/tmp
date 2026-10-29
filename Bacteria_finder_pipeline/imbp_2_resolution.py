import ast
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import posixpath
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


SOURCE = Path(__file__).resolve().parents[1] / "third_party/imbp_2/resolution/main.py"
FUNCTIONS = {"get_color_mask", "safe_add", "one_channel_processing", "process_image"}


@lru_cache(maxsize=1)
def _load_functions():
    try:
        source = ast.parse(SOURCE.read_text(encoding="utf-8-sig"), filename=str(SOURCE))
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
        if {node.name for node in functions} != FUNCTIONS:
            raise ValueError("В исходнике отсутствуют необходимые функции.")
        return compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec")
    except (OSError, SyntaxError, ValueError) as exc:
        raise RuntimeError(
            "Не удалось загрузить улучшение ×2 (imbp_2). "
            "Восстановите third_party/imbp_2/resolution/main.py из обновления."
        ) from exc


class _ImageStore:
    def __init__(self, checkpoint):
        self.files = {}
        self.checkpoint = checkpoint

    def imwrite(self, path, image):
        self.checkpoint()
        success, encoded = cv2.imencode(posixpath.splitext(path)[1], image)
        if not success:
            raise RuntimeError("Не удалось закодировать изображение при улучшении ×2 (imbp_2).")
        self.files[path] = encoded.tobytes()
        return True

    def imread(self, path):
        self.checkpoint()
        image = cv2.imdecode(np.frombuffer(self.files[path], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Не удалось декодировать изображение при улучшении ×2 (imbp_2).")
        return image

    def open(self, path):
        self.checkpoint()
        for channel in range(3):
            if path.endswith(f"_channel{channel}_ORIG.jpg"):
                self.checkpoint(f"Улучшение ×2 (imbp_2): канал {channel + 1} из 3…")
        with Image.open(BytesIO(self.files[path])) as image:
            return image.copy()


def enhance_image(image_bgr, checkpoint=lambda message="": None):
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size:
        raise ValueError("Для улучшения ×2 требуется непустое трёхканальное изображение BGR.")
    if image_bgr.dtype != np.uint8:
        raise ValueError("Для улучшения ×2 требуется изображение BGR типа uint8.")
    checkpoint("Улучшение ×2 (imbp_2): подготовка…")
    functions = _load_functions()
    store = _ImageStore(checkpoint)
    store.imwrite("input.png", image_bgr)
    store.files["test_images/input/input0_jpg.jpg"] = store.files.pop("input.png")
    namespace = {
        "np": np,
        "ndimage": ndimage,
        "cv2": SimpleNamespace(
            imread=store.imread, imwrite=store.imwrite, inRange=cv2.inRange,
            resize=cv2.resize, GaussianBlur=cv2.GaussianBlur, INTER_LANCZOS4=cv2.INTER_LANCZOS4,
        ),
        "Image": SimpleNamespace(open=store.open),
        "os": SimpleNamespace(path=posixpath, makedirs=lambda *args, **kwargs: None),
    }
    exec(functions, namespace)
    namespace["process_image"](0, "input", 2, 2)
    result = store.imread("test_images/input/Result_input0.jpg")
    if result.shape != (image_bgr.shape[0] * 2, image_bgr.shape[1] * 2, 3):
        raise RuntimeError("Улучшение ×2 (imbp_2) вернуло изображение неожиданного размера.")
    checkpoint()
    return result
