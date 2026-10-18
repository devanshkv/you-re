import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import your.writer as writer_module
from your import Your
from your.writer import Writer

_FIXTURE = Path(__file__).parent / "data" / "small.fil"


class _SilentProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        return None


@pytest.fixture
def your_object():
    with Your(str(_FIXTURE)) as reader:
        yield reader


def test_to_fil_accepts_arrays_and_reuses_the_output_path(your_object, tmp_path):
    data = your_object.get_data(0, 1)
    writer = Writer(
        your_object,
        nstart=0,
        nsamp=1,
        outdir=str(tmp_path),
        outname="repeat",
        progress=False,
    )
    output = tmp_path / "repeat.fil"

    writer.to_fil(data=data)
    assert output.is_file()
    assert writer.outname == "repeat"

    writer.to_fil(data=data)
    assert output.is_file()
    assert writer.outname == "repeat"
    assert not (tmp_path / "repeat.fil.fil").exists()
    with Your(str(output)) as result:
        assert result.your_header.nspectra == 1
        np.testing.assert_array_equal(result.get_data(0, 1), data)


def test_to_fil_joins_directory_without_mutating_name(your_object, tmp_path):
    writer = Writer(
        your_object, nsamp=2, outdir=str(tmp_path), outname="read", progress=False
    )
    writer.to_fil()
    writer.to_fil()
    assert writer.outname == "read"
    with Your(str(tmp_path / "read.fil")) as result:
        assert result.your_header.nspectra == 2
        np.testing.assert_array_equal(result.get_data(0, 2), your_object.get_data(0, 2))


@pytest.mark.parametrize("alias_kind", ("same_path", "symlink", "hardlink"))
def test_to_fil_rejects_input_aliases_without_truncating_source(tmp_path, alias_kind):
    source = tmp_path / "source.fil"
    shutil.copyfile(_FIXTURE, source)
    before = source.read_bytes()

    if alias_kind == "same_path":
        output_name = source.stem
    else:
        alias = tmp_path / "alias.fil"
        if alias_kind == "symlink":
            alias.symlink_to(source)
        else:
            os.link(source, alias)
        output_name = alias.stem

    with Your(str(source)) as source_reader:
        writer = Writer(
            source_reader,
            nsamp=0,
            gulp=1,
            outdir=str(tmp_path),
            outname=output_name,
            progress=False,
        )
        with pytest.raises(ValueError, match="must not overwrite its input"):
            writer.to_fil()

    assert source.read_bytes() == before


def test_to_fil_rejects_member_of_multi_file_input_without_truncating_source(
    tmp_path,
):
    fixture = Path(__file__).parent / "data" / "small.fits"
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fits"
    shutil.copyfile(fixture, first)
    shutil.copyfile(fixture, second)
    before = second.read_bytes()
    (tmp_path / "second.fil").symlink_to(second)

    with Your([str(first), str(second)]) as source_reader:
        writer = Writer(
            source_reader,
            nsamp=0,
            gulp=1,
            outdir=str(tmp_path),
            outname="second",
            progress=False,
        )
        with pytest.raises(ValueError, match="must not overwrite its input"):
            writer.to_fil()

    assert second.read_bytes() == before


def test_to_fil_writes_partial_final_gulp(your_object, tmp_path):
    writer = Writer(
        your_object,
        nstart=1,
        nsamp=5,
        gulp=3,
        outdir=str(tmp_path),
        outname="partial",
        progress=False,
    )

    writer.to_fil()

    with Your(str(tmp_path / "partial.fil")) as result:
        assert result.your_header.nspectra == 5
        np.testing.assert_array_equal(result.get_data(0, 5), your_object.get_data(1, 5))


@pytest.mark.parametrize("nsamp", (None, 20))
def test_to_fil_clips_requested_samples_to_input_end(your_object, tmp_path, nsamp):
    available = your_object.your_header.nspectra - 1
    requested = your_object.your_header.nspectra if nsamp is None else nsamp
    expected = min(requested, available)
    writer = Writer(
        your_object,
        nstart=1,
        nsamp=nsamp,
        gulp=2,
        outdir=str(tmp_path),
        outname="to_end",
        progress=False,
    )

    writer.to_fil()

    assert writer.nsamp == requested
    with Your(str(tmp_path / "to_end.fil")) as result:
        assert result.your_header.nspectra == expected
        np.testing.assert_array_equal(
            result.get_data(0, expected), your_object.get_data(1, expected)
        )


@pytest.mark.parametrize("gulp", (0, -1))
def test_to_fil_rejects_nonpositive_gulp_before_creating_output(
    your_object, tmp_path, gulp
):
    writer = Writer(
        your_object,
        nsamp=1,
        gulp=gulp,
        outdir=str(tmp_path),
        outname="invalid_gulp",
        progress=False,
    )

    with pytest.raises(ValueError, match="gulp must be positive"):
        writer.to_fil()

    assert not (tmp_path / "invalid_gulp.fil").exists()


def test_setup_dada_preserves_an_explicit_key(your_object, monkeypatch):
    created = []

    class FakeDadaManager:
        def __init__(self, size, key):
            self.size = size
            self.key = key
            self.setup_called = False
            created.append(self)

        def setup(self):
            self.setup_called = True
            return self

    dada_module = types.ModuleType("your.formats.dada")
    dada_module.DadaManager = FakeDadaManager
    monkeypatch.setitem(sys.modules, "your.formats.dada", dada_module)

    writer = Writer(your_object, gulp=4, progress=False)
    writer.setup_dada(dada_key="0x1234", data_step=2)

    assert writer.dada_key == "0x1234"
    assert writer.data_step == 2
    assert writer.dada_size == 2 * writer.nchans * np.dtype("uint8").itemsize
    assert writer.dada_is_set
    assert created[0].key == "0x1234"
    assert created[0].setup_called


def test_to_dada_marks_eod_on_the_last_full_page(your_object, monkeypatch):
    class RecordingDada:
        def __init__(self):
            self.events = []
            self.starts = []

        def dump_header(self, header):
            return None

        def dump_data(self, data):
            return None

        def mark_filled(self):
            self.events.append("filled")

        def eod(self):
            self.events.append("eod")

    writer = Writer(
        your_object,
        nstart=3,
        nsamp=5,
        gulp=2,
        progress=False,
    )
    dada = RecordingDada()
    writer.DM = dada
    writer.dada_is_set = True
    writer.data_step = 2

    def read_page(start, samples):
        dada.starts.append((start, samples))
        writer.data = np.zeros(
            (samples, 1, writer.nchans), dtype=your_object.your_header.dtype
        )
        return writer.data

    monkeypatch.setattr(writer, "get_data_to_write", read_page)
    monkeypatch.setattr(writer_module, "Progress", _SilentProgress)

    writer.to_dada()

    assert dada.starts == [(3, 2), (5, 2), (7, 2)]
    assert dada.events == ["filled", "filled", "eod"]
