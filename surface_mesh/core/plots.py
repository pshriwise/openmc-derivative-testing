"""Fixed-view OpenMC material slices and an incremental accepted-mesh animation."""

from pathlib import Path

from .runtime import atomic_write

ANIMATION_WIDTH = 1800
MATERIAL_COLORS = {'iron': (113, 128, 150), 'water': (50, 155, 209), 'air': (220, 70, 50)}


def save_solid_geometry_image(model, settings, output_path):
    """Render the initialized native geometry with the configured fixed camera."""
    import json
    import openmc.lib
    from PIL import Image

    plot = openmc.lib.SolidRayTracePlot()
    plot.pixels = settings.solid_pixels
    plot.color_by = plot.COLOR_BY_MATERIAL
    plot.set_default_colors()
    plot.set_all_opaque()  # Material mode leaves void transparent.
    for material in model.materials:
        plot.set_color(material.id, MATERIAL_COLORS.get(material.name, (170, 170, 170)))
    plot.camera_position = settings.solid_camera_position
    plot.look_at = settings.solid_look_at
    plot.up = settings.solid_up
    plot.fov = settings.box_solid_fov if settings.model_variant == 'box' else settings.solid_fov
    plot.light_position = settings.solid_camera_position
    plot.diffuse_fraction = settings.solid_diffuse_fraction
    plot.update_view()
    image = Image.fromarray(plot.create_image())
    try:
        atomic_write(output_path, lambda stream: image.save(stream, format='PNG'))
    finally:
        image.close()
    metadata = dict(renderer='openmc.lib.SolidRayTracePlot',
                    camera_position=plot.camera_position, look_at=plot.look_at,
                    up=plot.up, pixels=plot.pixels, fov=plot.fov,
                    light_position=plot.light_position, diffuse_fraction=plot.diffuse_fraction)
    encoded = (json.dumps(metadata, indent=2) + '\n').encode()
    atomic_write(Path(output_path).with_suffix('.json'), lambda stream: stream.write(encoded))


def save_geometry_image(model, mesh_path, settings, output_path):
    """Render the currently initialized OpenMC geometry before transport starts."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np

    figure = Figure(figsize=(12, 10), dpi=150, layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    colors = {material: MATERIAL_COLORS.get(material.name, (170, 170, 170))
              for material in model.materials}
    width = settings.geometry_detail_width
    mesh_cell = model.geometry.root_universe.cells[20]
    center = (tuple(mesh_cell.translation) if mesh_cell.translation is not None
              else (0., 0., 0.))
    source = np.asarray(model.settings.source[0].space.xyz)
    views = [('xy', (8., 0., 0.), (52., 34.), 'Iron, water, and source'),
             ('xy', center, (width, width), f'Iron — XY, z = {center[2]:g}'),
             ('xz', center, (width, width), f'Iron — XZ, y = {center[1]:g}'),
             ('yz', center, (width, width), f'Iron — YZ, x = {center[0]:g}')]
    for axis, (basis, origin, extent, title) in zip(axes, views):
        model.plot(origin=origin, width=extent,
                   pixels=(settings.geometry_plot_pixels, settings.geometry_plot_pixels),
                   basis=basis, color_by='material', colors=colors,
                   legend=False, axes=axis, n_samples=0)
        horizontal, vertical = ('xyz'.index(letter) for letter in basis)
        normal = 3 - horizontal - vertical
        if np.isclose(source[normal], origin[normal], rtol=0, atol=1e-10):
            axis.scatter([source[horizontal]], [source[vertical]], marker='x',
                         color='#c62828', s=35, zorder=3)
        axis.set_title(title)
        axis.set_xlabel(f'{basis[0]} [cm]')
        axis.set_ylabel(f'{basis[1]} [cm]')
    figure.suptitle(f'OpenMC material geometry: {Path(mesh_path).name}', fontsize=15)
    legend = [Patch(color=np.asarray(colors[material]) / 255, label=material.name.capitalize())
              for material in model.materials]
    legend.append(Line2D([], [], color='#c62828', marker='x', linestyle='', label='Neutron source'))
    figure.legend(handles=legend, loc='outside lower center', ncols=len(legend))
    atomic_write(output_path, lambda stream: figure.savefig(stream, format='png'))
    figure.clear()


def update_accepted_animation(root, history):
    """Combine slices, solid rendering, and objective history for accepted steps."""
    import io
    import json
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import font_manager
    from .report import accepted_evaluations, draw_accepted_history

    root = Path(root)
    width = ANIMATION_WIDTH
    column = width // 2
    height = round(width * 5 / 9)
    header = round(height * .05)
    render_height = round(height * .55)
    plot_height = height - header - render_height
    font = ImageFont.truetype(font_manager.findfont('DejaVu Sans'), max(12, width // 75))
    records = []
    rows = accepted_evaluations(history)
    for mode, filename in [('combined', 'accepted_geometry.gif'),
                           ('solid_image', 'accepted_geometry_solid.gif')]:
        frames = []
        try:
            for step, entry in rows:
                # A resample can lack image fields in an older archive; choose
                # the latest available image of this exact mesh in that case.
                images = {}
                for candidate in history['evaluations']:
                    if candidate['status'] == 'complete' and candidate['mesh'] == entry['mesh']:
                        for field in ('image', 'solid_image'):
                            if candidate.get(field) and (root / candidate[field]).is_file():
                                images[field] = candidate[field]
                if not images or (mode == 'solid_image' and 'solid_image' not in images):
                    continue
                if mode == 'solid_image':
                    with Image.open(root / images['solid_image']) as source:
                        frames.append(source.convert('RGB'))
                    continue
                frame = Image.new('RGB', (width, height), 'white')
                draw = ImageDraw.Draw(frame)
                draw.text((16, 8), f'Accepted update {step}  |  tally mean {entry["tally_mean"]:.7g}',
                          fill='#263238', font=font)
                for field, box in [('image', (0, header, column, height - header)),
                                   ('solid_image', (column, header, column, render_height))]:
                    x, y, tile_width, tile_height = box
                    if field not in images:
                        draw.text((x + 16, y + 16), f'{"3D render" if field == "solid_image" else "2D slices"} unavailable',
                                  fill='#607d8b', font=font)
                        continue
                    with Image.open(root / images[field]) as source:
                        tile = ImageOps.contain(source.convert('RGB'), (tile_width, tile_height),
                                                method=Image.Resampling.LANCZOS)
                        frame.paste(tile, (x + (tile_width - tile.width) // 2,
                                           y + (tile_height - tile.height) // 2))
                        tile.close()
                figure = Figure(figsize=(column / 100, plot_height / 100), dpi=100, layout='constrained')
                FigureCanvasAgg(figure)
                draw_accepted_history(figure.subplots(), history, current_step=step)
                with io.BytesIO() as buffer:
                    figure.savefig(buffer, format='png')
                    buffer.seek(0)
                    with Image.open(buffer) as chart:
                        frame.paste(chart.convert('RGB'), (column, header + render_height))
                figure.clear()
                frames.append(frame)
                records.append(dict(step=step, mesh=entry['mesh'], evaluation=entry['index'],
                                    tally_mean=entry['tally_mean'], tally_std_dev=entry['tally_std_dev'], **images))
            if frames:
                atomic_write(root / filename, lambda stream: frames[0].save(
                    stream, format='GIF', save_all=True, append_images=frames[1:],
                    duration=history['settings']['animation_duration_ms'], loop=0))
        finally:
            for frame in frames:
                frame.close()
    encoded = (json.dumps(records, indent=2) + '\n').encode()
    atomic_write(root / 'animation_frames.json', lambda stream: stream.write(encoded))
