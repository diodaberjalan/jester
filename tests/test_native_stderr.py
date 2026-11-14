"""Regression checks for filtering XLA's repetitive native warning."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_only_specific_native_warning_is_filtered_and_stderr_restored():
    result = _run(
        r"""
        import os, sys
        from jesterTOV._native_stderr import filter_xla_cost_model_warning
        with filter_xla_cost_model_warning():
            os.write(2, b"W1114 01:16:02.406731 11610 sol_gpu_cost_model.cc:102] No SoL config found for device: NVIDIA H200 NVL. Using default config.\n")
            os.write(2, b"W1201 01:16:02 10 sol_gpu_cost_model.cc:555] No SoL config found for device: Other GPU. Using default config.\n")
            os.write(2, b"W1114 unrelated warning\nE1114 important error\n")
            os.write(1, b"stdout unchanged\n")
        os.write(2, b"stderr restored\n")
        sys.exit(37)
        """
    )
    assert result.returncode == 37
    assert "No SoL config found" not in result.stderr
    assert "unrelated warning" in result.stderr
    assert "important error" in result.stderr
    assert "stderr restored" in result.stderr
    assert result.stdout == "stdout unchanged\n"


def test_stderr_is_restored_after_exception():
    result = _run(
        r"""
        import os
        from jesterTOV._native_stderr import filter_xla_cost_model_warning
        try:
            with filter_xla_cost_model_warning():
                os.write(2, b"before exception\n")
                raise ValueError("failure")
        except ValueError:
            os.write(2, b"after exception\n")
        """
    )
    assert result.returncode == 0
    assert "before exception" in result.stderr
    assert "after exception" in result.stderr


def test_inference_cli_applies_filter():
    result = _run(
        r"""
        import os, sys
        from jesterTOV.inference import run_inference
        def fake_main(config):
            assert config == "config.yaml"
            os.write(2, b"W1114 01:16:02 123 sol_gpu_cost_model.cc:102] No SoL config found for device: NVIDIA H200 NVL. Using default config.\n")
            os.write(2, b"E1114 actual failure\n")
        run_inference.main = fake_main
        sys.argv = ["run_jester_inference", "config.yaml"]
        run_inference.cli_entry_point()
        """
    )
    assert result.returncode == 0, result.stderr
    assert "No SoL config found" not in result.stderr
    assert "E1114 actual failure" in result.stderr
