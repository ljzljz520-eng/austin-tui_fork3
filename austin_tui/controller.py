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

import asyncio
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from enum import Enum
from pathlib import Path
from textwrap import wrap
from time import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from austin.aio import AsyncAustin
from austin.cli import AustinArgumentParser
from austin.cli import AustinCommandLineError
from austin.events import AustinMetadata
from austin.events import AustinSample
from austin.format.mojo import MojoStreamReader
from austin.format.mojo import MojoStreamWriter
from austin.stats import AustinStatsType
from psutil import Process

from austin_tui import AustinProfileMode
from austin_tui.adapters import Adapter
from austin_tui.adapters import CommandLineAdapter
from austin_tui.adapters import CountAdapter
from austin_tui.adapters import CpuAdapter
from austin_tui.adapters import CurrentThreadAdapter
from austin_tui.adapters import DurationAdapter
from austin_tui.adapters import FlameGraphAdapter
from austin_tui.adapters import MemoryAdapter
from austin_tui.adapters import ThreadDataAdapter
from austin_tui.adapters import ThreadFullDataAdapter
from austin_tui.adapters import ThreadNameAdapter
from austin_tui.adapters import ThreadTopDataAdapter
from austin_tui.model import FileLoadState
from austin_tui.model import Model
from austin_tui.view import ViewBuilder
from austin_tui.view.austin import AustinView
from austin_tui.view.austin import AustinViewMode
from austin_tui.widgets.markup import escape


class ThreadNav(Enum):
    """Thread navigation."""

    PREV = -1
    NEXT = 1


def _print(text: str) -> None:
    for line in wrap(text, 78):
        print(line, file=sys.stderr)


# MOJO "mode" metadata mapped to the TUI profile mode and stats container type.
_MOJO_MODES: Dict[str, Tuple[AustinProfileMode, AustinStatsType]] = {
    "wall": (AustinProfileMode.TIME, AustinStatsType.WALL),
    "cpu": (AustinProfileMode.TIME, AustinStatsType.CPU),
    "memory": (AustinProfileMode.MEMORY, AustinStatsType.MEMORY_ALLOC),
}

# Decode-worker batch budgets.
_BATCH_EVENTS = 512
_BATCH_BYTES = 1 << 18  # 256 KiB

# Queue protocol sentinels.
_EOF = object()
_CANCEL = object()


class _LoadError:
    """A decode failure reported by the worker thread."""

    __slots__ = ("error",)

    def __init__(self, error: Exception) -> None:
        self.error = error


def _decode_worker(
    path: str,
    queue: "asyncio.Queue[Any]",
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
) -> None:
    """Decode a MOJO file in a worker thread.

    The stream is read sequentially and converted into batches of Austin
    events that are posted back to the main event loop. Batches are flushed
    on an event-count or byte budget to keep UI updates frequent without
    flooding the loop. Any failure (truncation, corruption, I/O error) is
    posted as a :class:`_LoadError`.
    """

    def _post(item: Any) -> bool:
        """Post an item with back-pressure.

        Returns False if cancellation was requested or the loop is gone.
        """
        while not cancel_event.is_set():
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            try:
                future.result(timeout=0.2)
                return True
            except FutureTimeoutError:
                continue
            except Exception:
                # Event loop closed or stopping: nothing to deliver to.
                return False
        return False

    try:
        with open(path, "rb") as stream:
            reader = MojoStreamReader(stream)
            batch: List[Any] = []
            last_offset = 0
            for event in reader:
                if cancel_event.is_set():
                    return

                batch.append(event)
                offset = reader._offset
                if (
                    len(batch) >= _BATCH_EVENTS
                    or offset - last_offset >= _BATCH_BYTES
                ):
                    if not _post({"events": batch, "offset": offset}):
                        return
                    batch = []
                    last_offset = offset

            if batch:
                if not _post({"events": batch, "offset": reader._offset}):
                    return
            _post(_EOF)
    except Exception as exc:
        # MojoParseError, ValueError (bad header/unknown events), OSError, ...
        _post(_LoadError(exc))


class AustinTUIArgumentParser(AustinArgumentParser):
    """Austin TUI implementation of the Austin argument parser."""

    def __init__(self) -> None:
        super().__init__(name="austin-tui", full=False)

        self.add_argument(
            "-o",
            "--open",
            help="Open a MOJO file",
            type=Path,
        )

    def parse_args(self) -> Any:
        """Parse command line arguments and report any errors."""
        try:
            return super().parse_args()
        except AustinCommandLineError as e:
            reason, *code = e.args
            # If --open was given, the PID/command requirement doesn't apply.
            args, _ = super().parse_known_args()
            if getattr(args, "open", None) is not None:
                return args
            if reason:
                _print(reason)
            exit(code[0] if code else -1)


class AustinTUIController:
    """Austin controller.

    This controller is in charge of Austin data managing and UI updates.
    """

    model = Model.get()  # type: ignore[assignment]

    cpu = CpuAdapter
    memory = MemoryAdapter
    duration = DurationAdapter
    samples = CountAdapter
    current_thread = CurrentThreadAdapter
    thread_name = ThreadNameAdapter
    thread_data = ThreadDataAdapter
    thread_full_data = ThreadFullDataAdapter
    thread_top_data = ThreadTopDataAdapter
    command_line = CommandLineAdapter
    flamegraph = FlameGraphAdapter

    def __init__(self) -> None:
        self._view_mode = AustinViewMode.LIVE
        self._scaler: Optional[Callable[..., Any]] = None
        self._formatter: Optional[Callable[..., Any]] = None
        self._last_timestamp = 0
        self._update_task: Optional[asyncio.Task[None]] = None
        self._exception: Optional[Exception] = None
        self._file_mode = False

        self._cancel_event: Optional[threading.Event] = None
        self._load_queue: Optional[asyncio.Queue[Any]] = None

        view_builder = ViewBuilder.from_resource(
            "austin_tui.view", "tui.austinui"
        )

        self.austin: Optional[AsyncAustin] = None
        self.view: AustinView = view_builder.build()  # type: ignore[assignment]
        view = self.view
        self.view.callback = self.on_view_event

        view_builder.autoconnect(self)

        self.model.austin.mode = view.mode

        # Auto-create adapters
        for name, adapter_class in (
            (n, v)
            for n, v in type(self).__dict__.items()
            if isinstance(v, type) and v.__mro__[-2] == Adapter
        ):
            setattr(self, name, adapter_class(self.model, self.view))

    def set_thread_data(self) -> None:
        """Set the thread stack."""
        if not self.model.austin.threads:
            return

        if self._view_mode is AustinViewMode.GRAPH:
            self.flamegraph()  # type: ignore[call-arg]
        elif self._view_mode is AustinViewMode.FULL:
            self.thread_full_data()  # type: ignore[call-arg]
        elif self._view_mode is AustinViewMode.TOP:
            self.thread_top_data()  # type: ignore[call-arg]
        else:
            self.thread_data()  # type: ignore[call-arg]

        # self._last_timestamp = self.model.austin.stats.timestamp

    def set_thread(self) -> bool:
        """Set the thread to display."""
        self.current_thread()  # type: ignore[call-arg]
        self.thread_name()

        if not self.model.austin.threads:
            return True

        # Populate the thread stack view
        self.set_thread_data()

        return True

    def _add_flamegraph_palette(self) -> None:
        colors = [196, 202, 214, 124, 160, 166, 208]
        palette = self.view.palette

        for i, color in enumerate(colors):
            palette.add_color(f"fg{i}", 15, color)
            palette.add_color(f"fgf{i}", color)

        self.view.flamegraph.set_palette(
            (
                [palette.get_color(f"fg{i}") for i in range(len(colors))],
                [palette.get_color(f"fgf{i}") for i in range(len(colors))],
            )
        )

    async def start(self, args: Sequence[str]) -> None:
        """Start event."""
        pargs = AustinTUIArgumentParser().parse_args()  # type: ignore[call-arg]

        if pargs.open is not None:
            await self.open_file(pargs.open)
            return

        self.austin = AsyncAustin(
            self.on_sample, self.on_metadata, self.on_terminate
        )

        await self.austin.start(args)

        if pargs.pid is not None:
            child_process = Process(pargs.pid)
        else:
            austin_process = Process(self.austin._proc.pid)
            (child_process,) = austin_process.children()
        command = child_process.cmdline()

        mode = (
            AustinProfileMode.MEMORY if pargs.memory else AustinProfileMode.TIME
        )
        self.view.mode = mode

        """Austin ready callback."""
        self.model.system.set_child_process(child_process)
        # self.model.austin.set_metadata(self._meta)
        self.model.austin.set_command_line(command)

        self._add_flamegraph_palette()
        self.view.open()
        self._update_task = asyncio.create_task(self.update_loop())

        self._formatter, self._scaler = (
            (self.view.fmt_mem, self.view.scale_memory)
            if self.view.mode == AustinProfileMode.MEMORY
            else (self.view.fmt_time, self.view.scale_time)
        )
        self.model.system.start()

        self.command_line()

        self.view.set_pid(child_process.pid, pargs.children)

        try:
            await self.austin.wait()
        except Exception:
            self.shutdown()
            raise

        try:
            if self.view._input_task is not None:
                await self.view._input_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.shutdown()
            raise

        if self._exception is not None:
            raise self._exception

    async def open_file(self, path: Path) -> None:
        """Open a MOJO file with progressive, cancellable loading.

        The view is opened immediately against an empty staging model. The
        decoder runs in a worker thread and publishes revisions that the UI
        renders as they arrive. On EOF the staging model is validated and
        atomically committed. On error or cancellation the staging model is
        discarded and the previous snapshot (or an empty session) is
        restored.
        """
        self._file_mode = True
        self._view_mode = AustinViewMode.FULL
        self.view.file_mode = True
        self._cancel_event = threading.Event()

        try:
            total_bytes = path.stat().st_size if path.is_file() else 0
        except OSError:
            total_bytes = 0

        self.model.begin_file_load(path, total_bytes)

        # Open the view straight away so that loading progress is visible.
        self._add_flamegraph_palette()
        self.view.open()
        self.view.on_mode_selected(AustinViewMode.FULL)
        self.view.live_mode_cmd.set_color("disabled")
        self.view.save_cmd.set_color("disabled")
        self.view.play_pause_cmd.set_color("disabled")

        self._formatter, self._scaler = (
            self.view.fmt_time,
            self.view.scale_time,
        )
        self._update_task = asyncio.create_task(self.update_loop())

        self._render_load_progress()

        state, error = await self._load_file(path)

        await self._finish_load(state, error)

        try:
            if self.view._input_task is not None:
                await self.view._input_task
        except asyncio.CancelledError:
            pass

    async def _load_file(
        self, path: Path
    ) -> Tuple[FileLoadState, Optional[Exception]]:
        """Validate the path and run the decode worker and batch consumer."""
        if not path.exists():
            return (
                FileLoadState.FAILED,
                FileNotFoundError(f"No such MOJO file: '{path}'"),
            )
        if not path.is_file():
            return (
                FileLoadState.FAILED,
                OSError(f"Not a regular file: '{path}'"),
            )

        assert self._cancel_event is not None
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=8)
        self._load_queue = queue

        worker = loop.run_in_executor(
            None,
            _decode_worker,
            str(path),
            queue,
            loop,
            self._cancel_event,
        )

        try:
            return await self._consume_batches(queue)
        finally:
            self._load_queue = None

            if not worker.done():
                # The consumer bailed out before the worker finished: stop it.
                assert self._cancel_event is not None
                self._cancel_event.set()

            try:
                await asyncio.wait_for(worker, timeout=2.0)
            except Exception:
                pass

    async def _consume_batches(
        self, queue: "asyncio.Queue[Any]"
    ) -> Tuple[FileLoadState, Optional[Exception]]:
        """Apply published batches to the staging model.

        All model mutation happens here, on the main event-loop thread.
        """
        try:
            while True:
                item = await queue.get()

                assert self._cancel_event is not None
                if self._cancel_event.is_set() or item is _CANCEL:
                    return FileLoadState.CANCELLED, None

                if item is _EOF:
                    if self.model.austin.mode is None:
                        raise ValueError(
                            "missing 'mode' metadata: the MOJO file is "
                            "incompatible or corrupt"
                        )
                    return FileLoadState.READY, None

                if isinstance(item, _LoadError):
                    return FileLoadState.FAILED, item.error

                for event in item["events"]:
                    self._apply_event(event)

                self.model.publish_revision(item["offset"])
                self._render_load_progress()
        except Exception as exc:
            return FileLoadState.FAILED, exc

    def _apply_event(self, event: Any) -> None:
        """Apply a single Austin event to the staging model."""
        if isinstance(event, AustinMetadata):
            self._apply_metadata(event)
        elif isinstance(event, AustinSample):
            austin = self.model.austin
            if austin.mode is None:
                raise ValueError(
                    "MOJO stream contains samples before 'mode' metadata: "
                    "the file is incompatible or corrupt"
                )
            self._validate_sample(event)
            austin.update(event)

    def _validate_sample(self, sample: AustinSample) -> None:
        """Reject incomplete samples produced by a truncated stream."""
        if self.model.austin.mode is AustinProfileMode.MEMORY:
            if sample.metrics.memory is None:
                raise ValueError(
                    "truncated MOJO stream: incomplete trailing sample "
                    "(missing memory metric)"
                )
        elif sample.metrics.time is None:
            raise ValueError(
                "truncated MOJO stream: incomplete trailing sample "
                "(missing time metric)"
            )

    def _apply_metadata(self, metadata: AustinMetadata) -> None:
        """Configure the (staging) model from metadata, before samples."""
        name, value = metadata.name, metadata.value

        self.model.austin.add_metadata(name, value)

        if name == "mode":
            try:
                profile_mode, stats_type = _MOJO_MODES[value]
            except KeyError:
                raise ValueError(
                    f"incompatible MOJO 'mode' metadata: {value!r}"
                ) from None

            self.model.austin.set_mode(profile_mode, stats_type)
            self.view.mode = profile_mode
            self.view.set_mode(value)
            self._formatter, self._scaler = (
                (self.view.fmt_mem, self.view.scale_memory)
                if profile_mode is AustinProfileMode.MEMORY
                else (self.view.fmt_time, self.view.scale_time)
            )
        elif name == "python":
            self.view.set_python(value)
        elif name == "duration":
            self.model.system._duration = int(value) / 1e6

        # Austin/Python versions (compatibility information).
        austin_version, python_version = self.model.austin.get_versions()
        if name == "austin":
            austin_version = value
        if name == "python":
            python_version = value
        if austin_version and python_version:
            self.model.austin.set_versions(austin_version, python_version)

    def _render_load_progress(self) -> None:
        """Render the current staging revision and load progress."""
        model = self.model
        path = model.file_path
        name = path.name if path is not None else ""
        pct = model.file_progress * 100
        samples = model.austin.samples_count

        model_refresh = self.update()

        self.view.notification.set_text(
            f"Loading {name} {pct:5.1f}%  {samples} samples  (C) cancel"
        )

        if self._view_mode is AustinViewMode.GRAPH:
            self.view.flamegraph.draw()
        elif model_refresh:
            self.view.table.draw()

        if self.view.root_widget is not None:
            self.view.root_widget.refresh()

    async def _finish_load(
        self, state: FileLoadState, error: Optional[Exception]
    ) -> None:
        """Commit or roll back the load, then stop the live view behaviour."""
        if state is FileLoadState.READY:
            self.model.austin.set_command_line(
                ["<MOJO file>", str(self.model.file_path)]
            )
            self.model.commit_file_load()
            self.command_line()
            self.update()
        else:
            self.model.abort_file_load(state, error)

        # No more data is expected: cancel the update task and mark stopped.
        await self.stop()

        if state is FileLoadState.FAILED:
            self.view.notification.set_text(
                f"❌ Failed to open MOJO file: {error}"
            )
            self.view.notification.set_color("stopped")
        elif state is FileLoadState.CANCELLED:
            self.view.notification.set_text("Loading cancelled")
            self.view.notification.set_color("notify")

        if state is not FileLoadState.READY:
            # Force a full redraw so that stale staging content is replaced
            # by the restored snapshot (update() alone would not flag it).
            self.set_thread()

            # Labels that currently show staging statistics must be refreshed
            # against the restored snapshot.
            self.samples()
            self.duration()
            self.cpu()  # type: ignore[call-arg]
            self.memory()  # type: ignore[call-arg]

            if self._view_mode is AustinViewMode.GRAPH:
                self.view.flamegraph.draw()
            else:
                if not self.model.austin.threads:
                    self.view.table.set_data([])
                self.view.table.draw()

        if self.view.root_widget is not None:
            self.view.root_widget.refresh()

    async def stop(self) -> None:
        """Called when Austin exits: cancel the update task and mark the view stopped.

        Does not close the view — the user can still review final stats and press Q.
        """
        self.model.system.stop()

        if self._update_task is not None:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._exception = exc
            self._update_task = None

        self.view.stop()

    def update(self) -> bool:
        """Update event."""
        if self.model.frozen:
            return False

        # System data
        self.duration()
        self.cpu()  # type: ignore[call-arg]
        self.memory()  # type: ignore[call-arg]

        # Samples count
        self.samples()

        if self.model.austin.stats.timestamp > self._last_timestamp:
            return self.set_thread()

        return False

    async def update_loop(self) -> None:
        """The UI update loop."""
        try:
            while (
                not self.view._stopped
                and self.view.is_open
                and self.view.root_widget
            ):
                if self.update():
                    if self._view_mode is AustinViewMode.GRAPH:
                        self.view.flamegraph.draw()
                    else:
                        self.view.table.draw()

                self.view.root_widget.refresh()

                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break
        except Exception as exc:
            self.view.on_exception(exc)

    def _change_thread(self, direction: ThreadNav) -> bool:
        """Change thread."""
        austin = (
            self.model.frozen_austin or self.model.austin
            if self.model.frozen
            else self.model.austin
        )
        prev_index = austin.current_thread

        austin.current_thread = max(
            0,
            min(
                austin.current_thread + direction.value,
                len(austin.threads) - 1,
            ),
        )

        if prev_index != austin.current_thread:
            return self.set_thread()

        return False

    async def on_next_thread(self) -> bool:
        """Handle next thread event."""
        if self._change_thread(ThreadNav.NEXT):
            if self._view_mode is AustinViewMode.GRAPH:
                self.view.flamegraph.draw()
                self.view.flame_view.refresh()
            else:
                self.view.table.draw()
                self.view.stats_view.refresh()
            return True
        return False

    async def on_previous_thread(self) -> bool:
        """Handle previous thread event."""
        if self._change_thread(ThreadNav.PREV):
            if self._view_mode is AustinViewMode.GRAPH:
                self.view.flamegraph.draw()
                self.view.flame_view.refresh()
            else:
                self.view.table.draw()
                self.view.stats_view.refresh()
            return True
        return False

    async def on_live_mode_selected(self, _: Any = None) -> bool:
        """Select live mode."""
        if self._file_mode or self._view_mode is AustinViewMode.LIVE:
            return False

        self._view_mode = AustinViewMode.LIVE
        self.view.dataview_selector.select(0)
        self.set_thread_data()

        self.view.table.draw()
        self.view.stats_view.refresh()

        return True

    async def on_top_mode_selected(self, _: Any = None) -> bool:
        """Select top mode."""
        if self._view_mode is AustinViewMode.TOP:
            return False

        self._view_mode = AustinViewMode.TOP
        self.view.dataview_selector.select(0)
        self.set_thread_data()

        self.view.table.draw()
        self.view.stats_view.refresh()

        return True

    async def on_full_mode_selected(self, _: Any = None) -> bool:
        """Toggle full mode."""
        if self._view_mode is AustinViewMode.FULL:
            return False

        self._view_mode = AustinViewMode.FULL
        self.view.dataview_selector.select(0)
        self.set_thread_data()

        self.view.table.draw()
        self.view.stats_view.refresh()

        return True

    async def on_save(self, _: Any = None) -> bool:
        """Save the collected stats."""
        if self._file_mode:
            self.view.notification.set_text("")
            return False
        model = (
            self.model.frozen_austin if self.model.frozen else self.model.austin
        )

        def _dump_stats() -> None:
            assert self.model.system.child_process is not None
            pid = self.model.system.child_process.pid
            output_file = Path(f"austin_{int(time())}_{pid}").with_suffix(
                ".mojo"
            )
            try:
                with output_file.open("wb") as stream:
                    mojo_writer = MojoStreamWriter(stream)
                    for k, v in model.metadata.items():
                        mojo_writer.write(AustinMetadata(k, v))
                    for event in model.stats.flatten():
                        mojo_writer.write(event)
                self.view.notification.set_text(
                    self.view.markup(
                        f"Stats saved as <running>{escape(str(output_file))}</running> "
                    )
                )
            except IOError as e:
                self.view.notification.set_text(f"Failed to save stats: {e}")

            self.view.root_widget.refresh()

        await asyncio.get_event_loop().run_in_executor(None, _dump_stats)

        return False

    async def on_play_pause(self, _: Any = None) -> bool:
        """On play/pause handler."""
        if self.view._stopped or self._file_mode:
            return False

        self.model.toggle_freeze()
        self.update()
        self.view.notification.set_text(
            "Paused" if self.model.frozen else "Resumed"
        )
        return True

    async def on_cancel_load(self, _: Any = None) -> bool:
        """Cancel an in-progress file load."""
        if (
            self.model.file_state is not FileLoadState.LOADING
            or self._cancel_event is None
        ):
            return False

        self._cancel_event.set()
        if self._load_queue is not None:
            try:
                self._load_queue.put_nowait(_CANCEL)
            except asyncio.QueueFull:
                # Consumer will notice the cancel flag at the next batch.
                pass

        return False

    def _change_threshold(self, delta: float) -> float:
        self.model.austin.threshold += delta

        if self.model.austin.threshold < 0.0:
            self.model.austin.threshold = 0.0
        elif self.model.austin.threshold > 1.0:
            self.model.austin.threshold = 1.0

        if self.view._stopped or self.model.frozen:
            self.set_thread_data()
            self.view.table.draw()
            self.view.table.refresh()

        return self.model.austin.threshold

    async def on_threshold_up(self, _: Any = None) -> bool:
        """Handle threshold up."""
        th = self._change_threshold(0.01) * 100.0
        self.view.threshold.set_text(f"{th:.0f}%")
        return True

    async def on_threshold_down(self, _: Any = None) -> bool:
        """Handle threshold down."""
        th = self._change_threshold(-0.01) * 100.0
        self.view.threshold.set_text(f"{th:.0f}%")
        return True

    async def on_graph_selected(self, _: Any = None) -> bool:
        """Select graph visualisation."""
        if self._view_mode is AustinViewMode.GRAPH:
            return False

        self._view_mode = AustinViewMode.GRAPH

        self.view.dataview_selector.select(1)

        self.flamegraph()  # type: ignore[call-arg]

        return True

    def shutdown(self) -> None:
        """Force quit: terminate Austin and close the view immediately."""
        try:
            if self._cancel_event is not None:
                self._cancel_event.set()
        except Exception:
            pass
        try:
            if self.austin is not None:
                self.austin.terminate()
        except Exception:
            pass
        try:
            self.view.close()
        except Exception:
            pass

    def on_shutdown(self, _: Any = None) -> None:
        """The shutdown view event handler."""
        self.shutdown()

    def on_exception(self, exc: Exception) -> None:
        """The exception view event handler."""
        self.shutdown()
        raise exc

    # Austin events

    async def on_sample(self, sample: AustinSample) -> None:
        """Austin sample received callback."""
        self.model.austin.update(sample)

    async def on_metadata(self, metadata: AustinMetadata) -> None:
        """Austin metadata received callback."""
        self._apply_metadata(metadata)

    async def on_terminate(self) -> None:
        """Austin terminate callback."""
        await self.stop()

    # View events

    def on_view_event(self, event: AustinView.Event, data: Any = None) -> None:
        """View events handler."""

        def _unhandled(_: Any) -> None:
            raise RuntimeError(f"Unhandled view event: {event}")

        {
            AustinView.Event.QUIT: self.on_shutdown,
            AustinView.Event.EXCEPTION: self.on_exception,
        }.get(event, _unhandled)(data)  # type: ignore[operator]
