"""Transport and atomic persistence helpers shared with the CSG workflow."""

import os
from pathlib import Path
import tempfile


def configure_trigger(model, tally_id, rel_err, max_factor, interval):
    """Apply the existing flux relative-error trigger to a supplied model."""
    import openmc
    tally = next(t for t in model.tallies if t.id == tally_id)
    trigger = openmc.Trigger('rel_err', rel_err)
    trigger.scores = tally.scores
    tally.triggers = [trigger]
    model.settings.trigger_active = True
    model.settings.trigger_max_batches = model.settings.batches * max_factor
    model.settings.trigger_batch_interval = interval


def atomic_write(path, writer):
    """Replace a history/checkpoint only after its temporary file is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
