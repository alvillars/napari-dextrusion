"""Run ``dextrusion`` steps as child processes and stream their output to the GUI."""

from __future__ import annotations

from qtpy.QtCore import QObject, QProcess, Signal


class JobRunner(QObject):
    """Runs command lines one after the other; stops at the first failure.

    Signals: ``line(str)`` for every output line (stdout and stderr merged), ``step(int)`` when a
    command starts, ``done(bool)`` at the end (True if every command succeeded).
    """

    line = Signal(str)
    step = Signal(int)
    done = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: list[list[str]] = []
        self._index = -1
        self._buffer = ""
        self._proc: QProcess | None = None
        self._stopped = False

    @property
    def running(self) -> bool:
        return self._proc is not None

    def run(self, commands: list[list[str]]) -> None:
        if self.running:
            raise RuntimeError("a job is already running")
        self._queue, self._index, self._stopped = list(commands), -1, False
        self._next()

    def stop(self) -> None:
        self._stopped = True
        if self._proc is not None:
            self._proc.kill()

    def _next(self) -> None:
        self._index += 1
        if self._index >= len(self._queue):
            self._finish(True)
            return
        argv = self._queue[self._index]
        self.line.emit("$ " + " ".join(a if " " not in a else repr(a) for a in argv[3:]))
        self.step.emit(self._index)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        proc.start(argv[0], argv[1:])

    def _read(self) -> None:
        if self._proc is None:
            return
        self._buffer += bytes(self._proc.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buffer = self._buffer.replace("\r", "\n").split("\n")
        for text in lines:
            if text.strip():
                self.line.emit(text)

    def _on_finished(self, code, status=None) -> None:
        self._read()
        if self._buffer.strip():
            self.line.emit(self._buffer)
        self._buffer = ""
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.deleteLater()
        if self._stopped:
            self.line.emit("stopped")
            self._finish(False)
        elif code != 0:
            self.line.emit(f"failed (exit code {code})")
            self._finish(False)
        else:
            self._next()

    def _on_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.line.emit("could not start the process")
            self._proc = None
            self._finish(False)

    def _finish(self, ok: bool) -> None:
        self._queue, self._index = [], -1
        self.done.emit(ok)
