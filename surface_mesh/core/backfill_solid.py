"""Add native solid renders to completed evaluations without rerunning transport."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .settings import IntRunSettings
from .plots import save_solid_geometry_image
from .runtime import atomic_write


def backfill(root):
    import openmc
    import openmc.lib
    from .models import build_model, model_source_path

    root = Path(root).resolve()
    history = json.loads((root / 'history.json').read_text())
    if history['status'] == 'running':
        raise ValueError('Backfill requires a stopped run to avoid concurrent history writes')
    if history.get('mode') != 'int_inspired':
        raise ValueError('Solid backfill requires an INT run')
    cfg = IntRunSettings(**history['settings'])
    provenance = json.loads((root / 'provenance.json').read_text())
    library = Path(openmc.lib._dll._name).resolve()
    renderer_hash = hashlib.sha256(library.read_bytes()).hexdigest()
    sources = {model_source_path(cfg): provenance['model_sha256'],
               library: renderer_hash}
    def check_sources():
        for path, expected in sources.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Cannot backfill with a changed model/library: {path}')
    check_sources()
    render_provenance = dict(renderer_library=str(library), renderer_library_sha256=renderer_hash,
                             transport_library_sha256=provenance['library_sha256'])
    (root / 'solid_render_provenance.json').write_text(json.dumps(render_provenance, indent=2) + '\n')
    if not os.environ.get('OPENMC_CROSS_SECTIONS'):
        fallback = Path.home() / 'nndc_hdf5/cross_sections.xml'
        if not fallback.exists():
            raise RuntimeError('Set OPENMC_CROSS_SECTIONS')
        os.environ['OPENMC_CROSS_SECTIONS'] = str(fallback)
    openmc.config['cross_sections'] = os.environ['OPENMC_CROSS_SECTIONS']
    rendered = {}
    for entry in history['evaluations']:
        if entry['status'] != 'complete':
            continue
        check_sources()
        output = root / entry['directory'] / 'geometry_solid.png'
        if entry['mesh'] in rendered:
            source = rendered[entry['mesh']]
            shutil.copyfile(source, output)
            shutil.copyfile(source.with_suffix('.json'), output.with_suffix('.json'))
        else:
            openmc.reset_auto_ids()
            model = build_model(cfg, root / entry['mesh'])
            old_cwd = Path.cwd()
            with tempfile.TemporaryDirectory(prefix='solid_render_') as temporary:
                try:
                    os.chdir(temporary)
                    model.export_to_model_xml()
                    with openmc.lib.run_in_memory(args=['-s', str(cfg.threads)]):
                        save_solid_geometry_image(model, cfg, output)
                finally:
                    os.chdir(old_cwd)
            rendered[entry['mesh']] = output
        check_sources()
        entry['solid_image'] = str(output.relative_to(root))
        encoded = (json.dumps(history, indent=2) + '\n').encode()
        atomic_write(root / 'history.json', lambda stream: stream.write(encoded))
        print(f'Rendered {root.name}/{entry["directory"]}', flush=True)
    from .plots import update_accepted_animation
    from .report import save_objective_history
    save_objective_history(root, history)
    update_accepted_animation(root, history)
    return len(rendered)
