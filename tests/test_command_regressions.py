import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _capture_runs(module, monkeypatch):
    calls = []

    def fake_run(argv, *args, **kwargs):
        calls.append((list(argv), args, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "subprocess", subprocess, raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _load_dada_module(monkeypatch):
    class FakeWriter:
        instances = []

        def __init__(self):
            self.connected_key = None
            self.disconnected = False
            self.__class__.instances.append(self)

        def connect(self, key):
            self.connected_key = key

        def disconnect(self):
            self.disconnected = True

    psrdada = types.ModuleType("psrdada")
    psrdada.Writer = FakeWriter
    monkeypatch.setitem(sys.modules, "psrdada", psrdada)
    spec = importlib.util.spec_from_file_location(
        "_test_dada", Path(__file__).parents[1] / "your" / "formats" / "dada.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, FakeWriter


def test_heimdall_runs_argv_with_correct_boolean_and_giant_rate_flags(monkeypatch):
    import your.utils.heimdall as heimdall

    calls = _capture_runs(heimdall, monkeypatch)
    system_calls = []
    monkeypatch.setattr(
        heimdall,
        "os",
        types.SimpleNamespace(system=system_calls.append),
    )

    manager = heimdall.HeimdallManager(
        filename="input file; echo unsafe",
        max_giant_rate=7,
        fswap=True,
        no_scrunching=True,
    )
    manager.run()

    assert not system_calls
    assert len(calls) == 1
    argv, args, kwargs = calls[0]
    assert not args
    assert kwargs["check"] is True
    assert argv[0] == "heimdall"
    assert argv[argv.index("-f") + 1] == "input file; echo unsafe"
    assert "-max_giant_rate" in argv
    assert argv[argv.index("-max_giant_rate") + 1] == "7"
    assert "-max_gaint_rate" not in argv
    assert "-no_scrunching" in argv
    fswap_index = argv.index("-fswap")
    assert fswap_index == len(argv) - 1 or argv[fswap_index + 1].startswith("-")
    assert "True" not in argv


def test_heimdall_omits_false_boolean_flags(monkeypatch):
    import your.utils.heimdall as heimdall

    calls = _capture_runs(heimdall, monkeypatch)
    monkeypatch.setattr(
        heimdall,
        "os",
        types.SimpleNamespace(system=lambda command: 0),
    )

    heimdall.HeimdallManager(
        filename="input.fil",
        fswap=False,
        no_scrunching=False,
        rfi_no_narrow=False,
        rfi_no_broad=False,
    ).run()

    argv = calls[0][0]
    assert "-fswap" not in argv
    assert "-no_scrunching" not in argv
    assert "-rfi_no_narrow" not in argv
    assert "-rfi_no_broad" not in argv


def test_dada_uses_checked_subprocess_commands(monkeypatch):
    dada, fake_writer = _load_dada_module(monkeypatch)
    calls = _capture_runs(dada, monkeypatch)
    system_calls = []
    monkeypatch.setattr(
        dada,
        "os",
        types.SimpleNamespace(system=system_calls.append),
    )

    manager = dada.DadaManager(size=128, key="0x1234", n_readers=2).setup()
    manager.teardown()

    assert not system_calls
    assert [call[0] for call in calls] == [
        ["dada_db", "-d", "-k", "0x1234"],
        ["dada_db", "-b", "128", "-k", "0x1234", "-r", "2", "-n", "8", "-l", "-p"],
        ["dada_db", "-d", "-k", "0x1234"],
    ]
    assert [call[2]["check"] for call in calls] == [False, True, True]
    assert fake_writer.instances[0].connected_key == int("0x1234", 16)
    assert fake_writer.instances[0].disconnected


@pytest.mark.parametrize("operation", ["heimdall", "dada_create", "dada_destroy"])
def test_required_command_failures_propagate(monkeypatch, operation):
    def fail(argv, **kwargs):
        if kwargs["check"]:
            raise subprocess.CalledProcessError(1, argv)
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fail)
    if operation == "heimdall":
        from your.utils.heimdall import HeimdallManager

        with pytest.raises(subprocess.CalledProcessError):
            HeimdallManager(filename="input.fil").run()
    else:
        dada, fake_writer = _load_dada_module(monkeypatch)
        manager = dada.DadaManager(size=128)
        if operation == "dada_create":
            with pytest.raises(subprocess.CalledProcessError):
                manager.setup()
            assert not fake_writer.instances
        else:
            manager.writer = fake_writer()
            with pytest.raises(subprocess.CalledProcessError):
                manager.teardown()
            assert manager.writer.disconnected
