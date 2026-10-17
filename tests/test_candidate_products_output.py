import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from your.candidate import Candidate
from your.utils.candidate_pipeline import make_plan

DATA = Path(__file__).parent / "data" / "28.fil"


def test_processed_file_preserves_dtype_sampling_and_plot_contract(
    tmp_path, monkeypatch
):
    with Candidate(str(DATA), dm=10, tcand=2.02888, width=8) as cand:
        cand.make_products(time_size=17, frequency_size=19, dmsteps=11)
        filename = cand.save_h5(out_dir=str(tmp_path))
        with h5py.File(filename) as saved:
            assert saved["data_freq_time"].shape == (17, 19)
            assert saved["data_dm_time"].shape == (11, 17)
            assert saved["data_freq_time"].dtype == np.float16
            assert saved["data_dm_time"].dtype == np.float16
            assert saved.attrs["ft_detrended"]
            assert saved.attrs["products_normalized"]
            assert saved.attrs["time_bin"] == 4
            assert saved.attrs["tsamp"] == cand.native_tsamp * 4
            np.testing.assert_array_equal(
                saved.attrs["dm_trials"], np.linspace(0, 20, 11)
            )
        import your.utils.plotter as plotter

        def no_second_detrend(*args, **kwargs):
            raise AssertionError("already processed file was detrended again")

        monkeypatch.setattr(plotter, "detrend", no_second_detrend)
        plotter.plot_h5(filename, save=False)


def test_finished_product_validation_rejects_wrong_dtype():
    with Candidate(str(DATA), dm=10, tcand=2.0, width=2) as cand:
        plan = make_plan(cand, time_size=3, frequency_size=4, dmsteps=5)
        with pytest.raises(ValueError, match="shape or dtype"):
            cand.set_processed_products(
                plan, np.zeros((3, 4), np.float32), np.zeros((5, 3), np.float32)
            )


def test_cpu_import_does_not_require_cupy_or_numba():
    script = """
import importlib.abc
import sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('cupy', 'cupyx', 'numba'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, NoGPU())
from your.candidate import Candidate
from your.utils.candidate_pipeline import make_plan
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_candidate_identity_distinguishes_inputs_and_full_precision(tmp_path):
    import shutil

    other = tmp_path / "other.fil"
    shutil.copyfile(DATA, other)
    ids = []
    for filename, kwargs in (
        (DATA, {}),
        (DATA, {"width": 8}),
        (DATA, {"label": 1}),
        (DATA, {"tcand": 2.000000001}),
        (other, {}),
    ):
        params = dict(dm=10, tcand=2.0, width=2, label=0)
        params.update(kwargs)
        with Candidate(str(filename), **params) as cand:
            ids.append(cand.id)
    assert len(set(ids)) == len(ids)
    with Candidate(str(DATA), dm=10, tcand=2.0, width=2, label=0) as cand:
        assert cand.id == ids[0]


def test_failed_hdf5_save_preserves_existing_file_and_removes_staging(
    tmp_path, monkeypatch
):
    with Candidate(str(DATA), dm=10, tcand=2.0, width=2) as cand:
        cand.make_products(time_size=9, frequency_size=8, dmsteps=4)
        destination = Path(cand.save_h5(out_dir=tmp_path))
        original = destination.read_bytes()
        create = h5py.Group.create_dataset

        def fail_on_dmt(self, name, *args, **kwargs):
            if name == "data_dm_time":
                raise OSError("simulated full disk")
            return create(self, name, *args, **kwargs)

        monkeypatch.setattr(h5py.Group, "create_dataset", fail_on_dmt)
        with pytest.raises(OSError, match="simulated full disk"):
            cand.save_h5(out_dir=tmp_path)
        assert destination.read_bytes() == original
        assert list(tmp_path.iterdir()) == [destination]
        with pytest.raises(OSError, match="simulated full disk"):
            cand.save_h5(out_dir=tmp_path, fnout="new-output.h5")
        assert not (tmp_path / "new-output.h5").exists()
        assert list(tmp_path.iterdir()) == [destination]


def test_mutated_kill_mask_refreshes_identity_and_preserves_both_products(tmp_path):
    with Candidate(str(DATA), dm=10, tcand=2.0, width=2) as cand:
        cand.make_products(time_size=9, frequency_size=8, dmsteps=4)
        original_id = cand.id
        first = Path(cand.save_h5(out_dir=tmp_path))
        original = first.read_bytes()
        cand.kill_mask = np.ones(cand.nchans, dtype=bool)
        cand.make_products(time_size=9, frequency_size=8, dmsteps=4)
        assert cand.id != original_id
        second = Path(cand.save_h5(out_dir=tmp_path))
        assert second != first
        assert first.read_bytes() == original
        assert len(list(tmp_path.glob("*.h5"))) == 2
        with h5py.File(second) as output:
            assert output.attrs["kill_mask"].all()
            assert not np.any(output["data_freq_time"][:])
