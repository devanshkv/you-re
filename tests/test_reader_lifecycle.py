import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from your import Your
from your.candidate import Candidate
from your.writer import Writer


DATA_DIR = Path(__file__).parent / "data"


def _handles(reader):
    if reader.format == "fil":
        return reader.fp, reader._mmdata
    return (reader.fits._file,)


def _fd(reader):
    if reader.format == "fil":
        return reader.fp.fileno()
    return reader.fits._file._file.fileno()


def _assert_fd_closed(fd):
    with pytest.raises((OSError, ValueError)):
        os.fstat(fd)


def _assert_open(reader):
    for handle in _handles(reader):
        assert not handle.closed


def _assert_closed(handles):
    for handle in handles:
        assert handle.closed


def _close_if_possible(reader):
    close = getattr(reader, "close", None)
    if close is not None:
        close()
        return
    if reader.format == "fil":
        reader._mmdata.close()
        reader.fp.close()
    else:
        reader.fits.close()


@pytest.fixture(params=("28.fil", "28.fits"))
def reader_path(request):
    return DATA_DIR / request.param


def test_context_manager_returns_self_and_closes(reader_path):
    reader = Your(str(reader_path))
    handles = _handles(reader)
    fd = _fd(reader)
    with reader as entered:
        assert entered is reader
        _assert_open(reader)
    _assert_closed(handles)
    _assert_fd_closed(fd)


def test_context_manager_closes_on_error_without_suppressing(reader_path):
    reader = Your(str(reader_path))
    handles = _handles(reader)
    fd = _fd(reader)
    with pytest.raises(RuntimeError, match="boom"):
        with reader:
            raise RuntimeError("boom")
    _assert_closed(handles)
    _assert_fd_closed(fd)


def test_close_is_idempotent(reader_path):
    reader = Your(str(reader_path))
    handles = _handles(reader)
    fd = _fd(reader)
    reader.close()
    reader.close()
    _assert_closed(handles)
    _assert_fd_closed(fd)


def test_get_data_after_close_raises_without_reopening(reader_path):
    reader = Your(str(reader_path))
    handles = _handles(reader)
    fd = _fd(reader)
    reader.close()
    with pytest.raises(ValueError):
        reader.get_data(0, 1)
    _assert_closed(handles)
    _assert_fd_closed(fd)


def test_data_returned_before_close_remains_usable(reader_path):
    reader = Your(str(reader_path))
    data = reader.get_data(0, 4)
    expected = data.copy()
    handles = _handles(reader)
    fd = _fd(reader)
    reader.close()
    np.testing.assert_array_equal(data, expected)
    _assert_closed(handles)
    _assert_fd_closed(fd)


def test_fits_frequencies_remain_available_after_close():
    reader = Your(str(DATA_DIR / "28.fits"))
    frequencies = reader.frequencies.copy()
    handles = _handles(reader)
    fd = _fd(reader)
    reader.close()
    np.testing.assert_array_equal(reader.freqs, frequencies)
    np.testing.assert_array_equal(reader.frequencies, frequencies)
    _assert_closed(handles)
    _assert_fd_closed(fd)


def test_candidate_inherits_reader_lifecycle(reader_path):
    reader = Candidate(str(reader_path), dm=0)
    handles = _handles(reader)
    fd = _fd(reader)
    with reader as entered:
        assert entered is reader
    _assert_closed(handles)
    _assert_fd_closed(fd)


@pytest.mark.parametrize("filename", ("small.fil", "small.fits"))
def test_writer_does_not_close_borrowed_reader(tmp_path, filename):
    reader = Your(str(DATA_DIR / filename))
    handles = _handles(reader)
    fd = _fd(reader)
    writer = Writer(
        reader,
        nsamp=1,
        gulp=1,
        outdir=str(tmp_path) + "/",
        outname="converted",
        progress=False,
    )
    writer.to_fil()
    _assert_open(reader)
    _close_if_possible(reader)
    _assert_closed(handles)
    _assert_fd_closed(fd)


@pytest.mark.parametrize("filename", ("28.fil", "28.fits"))
def test_your_constructor_failure_closes_backend(monkeypatch, filename):
    your_module = importlib.import_module("your.your")

    handles = []
    fds = []

    def fail_header(reader):
        handles.extend(_handles(reader))
        fds.append(_fd(reader))
        raise RuntimeError("header failure")

    monkeypatch.setattr(your_module, "Header", fail_header)
    reader = Your.__new__(Your)
    with pytest.raises(RuntimeError, match="header failure"):
        Your.__init__(reader, str(DATA_DIR / filename))
    _assert_closed(handles)
    _assert_fd_closed(fds[0])


@pytest.mark.parametrize("filename", ("28.fil", "28.fits"))
def test_candidate_constructor_failure_closes_backend(monkeypatch, filename):
    handles = []
    fds = []

    def fail_reset(self, **kwargs):
        handles.extend(_handles(self))
        fds.append(_fd(self))
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(Candidate, "_reset_candidate", fail_reset)
    reader = Candidate.__new__(Candidate)
    with pytest.raises(RuntimeError, match="candidate failure"):
        Candidate.__init__(reader, fp=str(DATA_DIR / filename))
    _assert_closed(handles)
    _assert_fd_closed(fds[0])


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="requires Linux procfs")
def test_fits_close_releases_fixture_descriptors():
    filename = str((DATA_DIR / "28.fits").resolve())

    def count_descriptors():
        count = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}") == filename:
                    count += 1
            except OSError:
                pass
        return count

    before = count_descriptors()
    reader = Your(filename)
    frequencies = reader.frequencies
    expected = frequencies.copy()
    assert count_descriptors() > before
    reader.close()
    assert count_descriptors() == before
    np.testing.assert_array_equal(frequencies, expected)
    np.testing.assert_array_equal(reader.frequencies, expected)
