"""Live worker output remains observable without losing failure diagnostics."""

import os
import sys

from ..core.progress import run_logged


def test_progress_streams_batches_and_retains_failure_log(tmp_path, capsys):
    code = ("print('ordinary diagnostic'); "
            "print('[INT progress] Rendering solid geometry ...', flush=True); "
            "print(' Simulating batch 1', flush=True); "
            "print('failure details'); raise SystemExit(7)")
    path = tmp_path / 'transport.log'
    with path.open('w') as log:
        result = run_logged([sys.executable, '-u', '-c', code], log,
                            cwd=tmp_path, env=os.environ.copy(), label='evaluation_0000')
    assert result == 7
    assert 'ordinary diagnostic' in path.read_text()
    assert 'failure details' in path.read_text()
    output = capsys.readouterr().out
    assert 'Rendering solid geometry' in output
    assert 'Simulating batch 1' in output
    assert 'ordinary diagnostic' not in output


def test_quiet_worker_reports_liveness(tmp_path, capsys):
    code = "import time; print('[INT progress] Rendering ...', flush=True); time.sleep(.15)"
    with (tmp_path / 'log').open('w') as log:
        assert run_logged([sys.executable, '-u', '-c', code], log, cwd=tmp_path,
                          env=os.environ.copy(), label='evaluation_0000', heartbeat=.02) == 0
    assert 'still running' in capsys.readouterr().out
