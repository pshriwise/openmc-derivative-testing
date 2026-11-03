"""Console progress for INT workers, with complete subprocess logs retained."""

from contextlib import contextmanager
from queue import Empty, Queue
import subprocess
from threading import Thread
from time import monotonic


@contextmanager
def phase(label, *, enabled=True):
    """Print a phase and its elapsed time without changing its execution."""
    start = monotonic()
    if enabled:
        print(f'[INT progress] {label} ...', flush=True)
    try:
        yield
    except BaseException:
        if enabled:
            print(f'[INT progress] {label} failed after {monotonic() - start:.1f}s', flush=True)
        raise
    else:
        if enabled:
            print(f'[INT progress] {label} completed in {monotonic() - start:.1f}s', flush=True)


def run_logged(command, log, *, cwd, env, label, heartbeat=30.):
    """Stream phase/batch messages and report liveness during quiet native calls."""
    start = monotonic()
    queue = Queue()
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors='replace') as process:
        def read_output():
            try:
                for line in process.stdout:
                    queue.put(line)
            finally:
                queue.put(None)

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        last_phase = 'worker startup'
        next_heartbeat = monotonic() + heartbeat
        try:
            while True:
                try:
                    line = queue.get(timeout=max(0., next_heartbeat - monotonic()))
                except Empty:
                    print(f'[INT progress] {label}: still running ({last_phase}); '
                          f'{monotonic() - start:.0f}s elapsed; see transport.log', flush=True)
                    next_heartbeat = monotonic() + heartbeat
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if '[INT progress]' in line or 'Simulating batch ' in line:
                    last_phase = line.strip().removeprefix('[INT progress] ')
                    print(f'[{label} +{monotonic() - start:.1f}s] {last_phase}', flush=True)
                    next_heartbeat = monotonic() + heartbeat
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            reader.join(timeout=5)
