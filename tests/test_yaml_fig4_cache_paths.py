from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _load_script33():
    spec_path = REPO / "scripts" / "33_run_yaml_fig4_with_split_plots.py"
    spec = importlib.util.spec_from_file_location("yaml_fig4_with_split_plots", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script33_forwards_live_and_force_profile_to_script30(monkeypatch):
    mod = _load_script33()
    commands = []

    def fake_run(cmd, print_command):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(mod, "_run", fake_run)

    rc = mod.main([
        "--no-split-plots",
        "--config",
        "configs/example.yaml",
        "--live",
        "--force-profile",
        "--run-name",
        "cache_path_test",
    ])

    assert rc == 0
    assert len(commands) == 1
    cmd30 = commands[0]
    assert str(mod.SCRIPT_30) in cmd30
    assert "--live" in cmd30
    assert "--force-profile" in cmd30
