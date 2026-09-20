# This file is part of "austin-tui" which is released under GPL.
#
# See file LICENCE or go to http://www.gnu.org/licenses/ for full license
# details.
#
# austin-tui is top-like TUI for Austin.
#
# Copyright (c) 2018-2020 Gabriele N. Tornetta <phoenix1987@gmail.com>.
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from enum import Enum
from pathlib import Path
from typing import Optional
from typing import Tuple

from austin_tui.model.austin import AustinModel
from austin_tui.model.system import FrozenSystemModel
from austin_tui.model.system import SystemModel


class FileLoadState(Enum):
    """File-load lifecycle states.

    These describe the state of a MOJO file being loaded into a staging
    model. The staging model is isolated from the committed model until the
    load is validated and atomically committed.
    """

    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Model:
    """The application model."""

    __slots__ = (
        "austin",
        "system",
        "frozen_austin",
        "frozen_system",
        "frozen",
        "file_state",
        "file_error",
        "file_path",
        "file_bytes_total",
        "file_bytes_read",
        "file_revision",
        "_previous_austin",
        "_previous_system",
    )

    _instance: Optional["Model"] = None

    @classmethod
    def get(cls) -> "Model":
        """Get the single model instance."""
        if cls._instance is not None:
            return cls._instance

        model = cls._instance = cls()
        return model

    def __init__(self) -> None:
        self.austin = AustinModel()
        self.system = SystemModel()
        self.frozen_austin: Optional[AustinModel] = None
        self.frozen_system: Optional[FrozenSystemModel] = None
        self.frozen = False

        self.file_state: Optional[FileLoadState] = None
        self.file_error: Optional[Exception] = None
        self.file_path: Optional[Path] = None
        self.file_bytes_total = 0
        self.file_bytes_read = 0
        self.file_revision = 0
        self._previous_austin: Optional[AustinModel] = None
        self._previous_system: Optional[SystemModel] = None

    def toggle_freeze(self) -> None:
        """Toggle the freeze status."""
        if self.frozen:
            self.unfreeze()
        else:
            self.freeze()

    def freeze(self) -> None:
        """Freeze the model."""
        self.frozen_austin = self.austin.freeze()
        self.frozen_system = self.system.freeze()
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the model."""
        self.frozen_austin = None
        self.frozen_system = None
        self.frozen = False

    # ---- File loading with isolated staging ----

    def begin_file_load(
        self, path: Path, total_bytes: int = 0
    ) -> Tuple[AustinModel, SystemModel]:
        """Begin loading a file into an isolated staging model.

        The currently committed models are set aside so that they can be
        restored if the load fails or is cancelled. The staging models are
        immediately exposed via :attr:`austin` and :attr:`system` so that the
        UI can render load progress as revisions are published.
        """
        self._previous_austin = self.austin
        self._previous_system = self.system

        self.austin = AustinModel()
        self.system = SystemModel()

        self.file_state = FileLoadState.LOADING
        self.file_error = None
        self.file_path = path
        self.file_bytes_total = total_bytes
        self.file_bytes_read = 0
        self.file_revision = 0

        return self.austin, self.system

    def publish_revision(self, bytes_read: int) -> None:
        """Publish a staging revision after a batch has been applied."""
        self.file_revision += 1
        self.file_bytes_read = bytes_read

    @property
    def file_progress(self) -> float:
        """The load progress as a fraction in [0, 1]."""
        if not self.file_bytes_total:
            return 0.0
        return min(1.0, self.file_bytes_read / self.file_bytes_total)

    def commit_file_load(self) -> None:
        """Atomically commit the staging models.

        Called after EOF and successful validation. The staging models (which
        are the currently exposed ones) become the committed models and the
        previous snapshot is discarded.
        """
        self.file_state = FileLoadState.READY
        self.file_bytes_read = self.file_bytes_total
        self.file_revision += 1

        self._previous_austin = None
        self._previous_system = None

    def abort_file_load(
        self,
        state: FileLoadState,
        error: Optional[Exception] = None,
    ) -> None:
        """Abort a file load.

        Discard the staging models and restore the previously committed
        snapshot (or an empty session when there was nothing to restore). The
        resulting state is either :attr:`FileLoadState.FAILED` or
        :attr:`FileLoadState.CANCELLED`.
        """
        assert state in (FileLoadState.FAILED, FileLoadState.CANCELLED)

        if self._previous_austin is not None:
            self.austin = self._previous_austin
            self.system = self._previous_system  # type: ignore[assignment]

        self.file_state = state
        self.file_error = error

        self._previous_austin = None
        self._previous_system = None
