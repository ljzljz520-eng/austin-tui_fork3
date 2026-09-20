import asyncio
import threading
from pathlib import Path

import pytest
from austin.events import AustinFrame
from austin.events import AustinMetadata
from austin.events import AustinMetrics
from austin.events import AustinSample
from austin.format.mojo import MojoStreamWriter

from austin_tui.controller import _CANCEL
from austin_tui.controller import _EOF
from austin_tui.controller import AustinTUIController
from austin_tui.controller import _decode_worker
from austin_tui.controller import _LoadError
from austin_tui.model import FileLoadState
from austin_tui.model import Model
from austin_tui.model.austin import AustinModel
from austin_tui.model.system import SystemModel
from austin_tui.view.austin import AustinViewMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_frame():
    return AustinFrame("test.py", "main", 1)


def make_sample(i=0, time=1000, memory=None):
    return AustinSample(
        pid=42,
        iid=0,
        thread="MainThread",
        metrics=AustinMetrics(time=time, memory=memory),
        frames=(make_frame(),),
    )


def make_samples(n):
    return [make_sample(i, 1000 * (i + 1)) for i in range(n)]


def standard_metadata(mode="wall", duration="1000000"):
    return [
        AustinMetadata("mode", mode),
        AustinMetadata("duration", duration),
        AustinMetadata("austin", "3.0"),
        AustinMetadata("python", "3.14"),
    ]


def write_mojo(path, events):
    with open(path, "wb") as stream:
        writer = MojoStreamWriter(stream)
        for event in events:
            writer.write(event)


def run_decode_worker(path, cancel_event=None):
    """Run the decode worker in a plain thread and collect queue items."""
    cancel_event = cancel_event or threading.Event()

    async def _collect():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        thread = threading.Thread(
            target=_decode_worker,
            args=(str(path), queue, loop, cancel_event),
        )
        thread.start()

        items = []
        while thread.is_alive() or not queue.empty():
            try:
                items.append(await asyncio.wait_for(queue.get(), timeout=0.5))
            except asyncio.TimeoutError:
                continue

        thread.join(2.0)
        while not queue.empty():
            items.append(queue.get_nowait())
        return items

    return asyncio.run(_collect())


@pytest.fixture
def controller():
    controller = AustinTUIController()
    controller._cancel_event = threading.Event()
    controller._file_mode = True
    yield controller

    # Reset the shared singleton so tests do not leak state into each other.
    model = controller.model
    model.austin = AustinModel()
    model.system = SystemModel()
    model.file_state = None
    model.file_error = None
    model.file_path = None
    model.file_bytes_total = 0
    model.file_bytes_read = 0
    model.file_revision = 0
    model._previous_austin = None
    model._previous_system = None


def drain(controller, items):
    """Feed queue items to the batch consumer."""

    async def _run():
        queue: asyncio.Queue = asyncio.Queue()
        for item in items:
            queue.put_nowait(item)
        return await controller._consume_batches(queue)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Model: staging lifecycle
# ---------------------------------------------------------------------------


def test_begin_file_load_sets_loading():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 100)

    assert model.file_state is FileLoadState.LOADING
    assert model.file_path == Path("x.mojo")
    assert model.file_bytes_total == 100
    assert model.file_bytes_read == 0
    assert model.file_revision == 0
    assert model.file_error is None


def test_begin_file_load_creates_isolated_staging():
    model = Model()
    old_austin = model.austin
    old_system = model.system

    model.begin_file_load(Path("x.mojo"), 100)

    assert model.austin is not old_austin
    assert model.system is not old_system
    assert model._previous_austin is old_austin
    assert model._previous_system is old_system
    # The new staging model is pristine.
    assert model.austin.samples_count == 0
    assert model.austin.metadata is None


def test_publish_revision_updates_revision_and_bytes():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 100)

    model.publish_revision(40)
    assert model.file_revision == 1
    assert model.file_bytes_read == 40

    model.publish_revision(80)
    assert model.file_revision == 2
    assert model.file_bytes_read == 80


def test_file_progress_fraction():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 200)

    model.publish_revision(50)
    assert model.file_progress == pytest.approx(0.25)


def test_file_progress_zero_without_total():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 0)

    model.publish_revision(50)
    assert model.file_progress == 0.0


def test_file_progress_clamped_to_one():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 100)

    model.publish_revision(150)
    assert model.file_progress == 1.0


def test_commit_file_load_ready_discards_previous():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 100)
    model.publish_revision(100)

    model.commit_file_load()

    assert model.file_state is FileLoadState.READY
    assert model.file_bytes_read == 100
    assert model._previous_austin is None
    assert model._previous_system is None


def test_abort_failed_restores_snapshot_and_error():
    model = Model()
    old_austin = model.austin
    old_system = model.system
    model.begin_file_load(Path("x.mojo"), 100)
    model.publish_revision(40)
    error = ValueError("boom")

    model.abort_file_load(FileLoadState.FAILED, error)

    assert model.file_state is FileLoadState.FAILED
    assert model.file_error is error
    assert model.austin is old_austin
    assert model.system is old_system


def test_abort_cancelled_restores_snapshot():
    model = Model()
    old_austin = model.austin
    old_system = model.system
    model.begin_file_load(Path("x.mojo"), 100)

    model.abort_file_load(FileLoadState.CANCELLED)

    assert model.file_state is FileLoadState.CANCELLED
    assert model.austin is old_austin
    assert model.system is old_system


def test_abort_rejects_invalid_states():
    model = Model()
    model.begin_file_load(Path("x.mojo"), 100)

    with pytest.raises(AssertionError):
        model.abort_file_load(FileLoadState.LOADING)

    with pytest.raises(AssertionError):
        model.abort_file_load(FileLoadState.READY)


# ---------------------------------------------------------------------------
# Decode worker
# ---------------------------------------------------------------------------


def test_decode_worker_valid_file_publishes_batches_and_eof(tmp_path):
    path = tmp_path / "good.mojo"
    write_mojo(path, standard_metadata() + make_samples(200))

    items = run_decode_worker(path)

    assert items[-1] is _EOF
    batches = items[:-1]
    assert batches
    assert all(isinstance(b["offset"], int) for b in batches)
    # The reader offset lags the final read, so it cannot exceed file size.
    assert batches[-1]["offset"] <= path.stat().st_size

    events = [e for b in batches for e in b["events"]]
    assert sum(isinstance(e, AustinSample) for e in events) == 200
    assert sum(isinstance(e, AustinMetadata) for e in events) == 4


def test_decode_worker_bad_header_publishes_load_error(tmp_path):
    path = tmp_path / "bad.mojo"
    path.write_bytes(b"not a mojo stream at all")

    items = run_decode_worker(path)

    assert len(items) == 1
    assert isinstance(items[0], _LoadError)


def test_decode_worker_corrupt_bytes_publishes_load_error(tmp_path):
    path = tmp_path / "corrupt.mojo"
    # More than one event-count budget so that a batch is published first.
    write_mojo(path, standard_metadata() + make_samples(600))
    with open(path, "ab") as stream:
        stream.write(b"\xff\xff\xff\xff")

    items = run_decode_worker(path)

    assert isinstance(items[-1], _LoadError)
    # The already-decoded prefix was published as a batch before the error.
    events = [e for b in items[:-1] for e in b["events"]]
    assert any(isinstance(e, AustinSample) for e in events)


def test_decode_worker_truncated_tail_publishes_load_error(tmp_path):
    path = tmp_path / "trunc.mojo"
    write_mojo(path, standard_metadata() + make_samples(200))

    # Cut the last 50 bytes: guaranteed to fall inside trailing events.
    size = path.stat().st_size
    path.write_bytes(path.read_bytes()[: size - 50])

    items = run_decode_worker(path)

    assert isinstance(items[-1], _LoadError)


def test_decode_worker_preset_cancel_stops_without_eof(tmp_path):
    path = tmp_path / "good.mojo"
    write_mojo(path, standard_metadata() + make_samples(200))
    cancel_event = threading.Event()
    cancel_event.set()

    items = run_decode_worker(path, cancel_event)

    assert items == []


# ---------------------------------------------------------------------------
# Batch consumer
# ---------------------------------------------------------------------------


def test_consume_batches_valid_events_ready(controller, tmp_path):
    path = tmp_path / "good.mojo"
    controller.model.begin_file_load(path, 400)

    state, error = drain(
        controller,
        [
            {"events": standard_metadata(), "offset": 100},
            {"events": make_samples(50), "offset": 300},
            _EOF,
        ],
    )

    assert state is FileLoadState.READY
    assert error is None
    assert controller.model.austin.samples_count == 50
    assert len(controller.model.austin.threads) == 1


def test_consume_batches_cancelled(controller, tmp_path):
    path = tmp_path / "good.mojo"
    controller.model.begin_file_load(path, 400)
    assert controller._cancel_event is not None
    controller._cancel_event.set()

    state, error = drain(
        controller,
        [{"events": make_samples(10), "offset": 100}],
    )

    assert state is FileLoadState.CANCELLED
    assert error is None


def test_consume_batches_cancel_sentinel(controller, tmp_path):
    path = tmp_path / "good.mojo"
    controller.model.begin_file_load(path, 400)

    state, _ = drain(controller, [_CANCEL])

    assert state is FileLoadState.CANCELLED


def test_consume_batches_bad_mode_metadata_failed(controller, tmp_path):
    path = tmp_path / "bad.mojo"
    controller.model.begin_file_load(path, 100)

    state, error = drain(
        controller,
        [{"events": [AustinMetadata("mode", "bogus")], "offset": 10}],
    )

    assert state is FileLoadState.FAILED
    assert isinstance(error, ValueError)
    assert "incompatible" in str(error)


def test_consume_batches_samples_before_mode_failed(controller, tmp_path):
    path = tmp_path / "bad.mojo"
    controller.model.begin_file_load(path, 100)

    state, error = drain(
        controller,
        [{"events": [make_sample()], "offset": 10}],
    )

    assert state is FileLoadState.FAILED
    assert isinstance(error, ValueError)
    assert "before 'mode' metadata" in str(error)


def test_consume_batches_incomplete_sample_failed(controller, tmp_path):
    path = tmp_path / "trunc.mojo"
    controller.model.begin_file_load(path, 100)

    incomplete = AustinSample(
        pid=42,
        iid=0,
        thread="MainThread",
        metrics=AustinMetrics(),
        frames=(make_frame(),),
    )

    state, error = drain(
        controller,
        [
            {"events": standard_metadata(), "offset": 50},
            {"events": [incomplete], "offset": 90},
        ],
    )

    assert state is FileLoadState.FAILED
    assert isinstance(error, ValueError)
    assert "truncated" in str(error)


def test_consume_batches_load_error_failed(controller, tmp_path):
    path = tmp_path / "bad.mojo"
    controller.model.begin_file_load(path, 100)
    failure = ValueError("parse error")

    state, error = drain(controller, [_LoadError(failure)])

    assert state is FileLoadState.FAILED
    assert error is failure


# ---------------------------------------------------------------------------
# Load flow
# ---------------------------------------------------------------------------


def test_load_file_missing_path_failed(controller, tmp_path):
    path = tmp_path / "missing.mojo"

    state, error = asyncio.run(controller._load_file(path))

    assert state is FileLoadState.FAILED
    assert isinstance(error, FileNotFoundError)
    assert controller._load_queue is None


def test_load_file_directory_failed(controller, tmp_path):
    state, error = asyncio.run(controller._load_file(tmp_path))

    assert state is FileLoadState.FAILED
    assert isinstance(error, OSError)


def test_load_file_valid_returns_ready(controller, tmp_path):
    path = tmp_path / "good.mojo"
    write_mojo(path, standard_metadata() + make_samples(200))
    controller.model.begin_file_load(path, path.stat().st_size)
    controller._view_mode = AustinViewMode.FULL

    state, error = asyncio.run(controller._load_file(path))

    assert state is FileLoadState.READY
    assert error is None
    assert controller.model.austin.samples_count == 200
