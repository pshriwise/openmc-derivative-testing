"""Check config-driven launchers without running OpenMC transport."""

from pathlib import Path

from ..cli import run as command
from ..core.settings import IntRunSettings
from ..int_optimizer import IntConfig


def test_run_config_overrides_reach_core(tmp_path, monkeypatch):
    optimizer = tmp_path / 'optimizer.json'
    optimizer.write_text('{"int_inspired": {"iterations": 3, "learning_rate": 0.02}}')
    settings = tmp_path / 'run.json'
    settings.write_text('{"int_run": {"particles": 200, "model_variant": "sphere"}}')
    output = tmp_path / 'output'
    seen = []

    def run(mesh, destination, int_config, cfg, **kwargs):
        seen.append((mesh, destination, int_config, cfg, kwargs))
        return {'status': 'completed', 'accepted_mesh': 'initial.h5m'}

    monkeypatch.setattr(command, 'run', run)
    assert command.main([
        '--input', 'input.h5m', '--output', str(output),
        '--int-config', str(optimizer), '--run-config', str(settings),
        '--particles', '400', '--model', 'box', '--iterations', '2',
        '--inner-iterations', '1', '--no-adaptive-sampling',
    ]) == 0
    mesh, destination, int_config, cfg, kwargs = seen.pop()
    assert mesh == Path('input.h5m') and destination == output
    assert isinstance(int_config, IntConfig) and isinstance(cfg, IntRunSettings)
    assert int_config.iterations == 1 and int_config.learning_rate == 0.02
    assert cfg.particles == 400 and cfg.model_variant == 'box' and cfg.max_iterations == 2
    assert not cfg.adaptive_sampling.enabled
    assert kwargs == {'continuation': None, 'allow_changed_model': False}


def test_run_finds_package_presets(tmp_path, monkeypatch):
    seen = []

    def run(mesh, destination, int_config, cfg, **kwargs):
        seen.append((mesh, int_config, cfg))
        return {'status': 'completed', 'accepted_mesh': 'initial.h5m'}

    monkeypatch.setattr(command, 'run', run)
    assert command.main(['--output', str(tmp_path / 'run')]) == 0
    mesh, optimizer, settings = seen.pop()
    assert mesh.is_file() and mesh.name == 'sphere.h5m'
    assert optimizer.volume_weight == 5.0
    assert settings.particles == 10000
