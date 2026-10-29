from traceback import format_exc

from PyQt5 import QtCore

from .models import PipelineSettings
from .pipeline import ProcessingCancelled, run_pipeline


class ProcessingWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    result_ready = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, image_bgr, settings: PipelineSettings, parent=None):
        super().__init__(parent)
        self._image_bgr = image_bgr.copy()
        self._settings = settings

    def run(self):
        try:
            result = run_pipeline(
                self._image_bgr,
                self._settings,
                on_progress=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except ProcessingCancelled:
            pass
        except Exception as exc:
            self.failed.emit(str(exc), format_exc())
