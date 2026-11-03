"""Box-based CSG test with the existing iron mesh and a visible air source cell."""

from pathlib import Path

import numpy as np
import openmc

from .model_defaults import BOX_CONTAINER_LOWER_CM, BOX_CONTAINER_UPPER_CM

INPUT_MESH = Path(__file__).resolve().parents[1] / 'results' / 'inputs/sphere_radius_2cm_fine.h5m'
IRON_CENTER_CM = (10.0, 0.0, 0.0)
WATER_LOWER_CM = (20.0, -5.0, -5.0)
WATER_UPPER_CM = (30.0, 5.0, 5.0)
WORLD_LOWER_CM = (-35.0, -35.0, -35.0)
WORLD_UPPER_CM = (35.0, 35.0, 35.0)
SOURCE_CENTER_CM = (0.0, 0.0, 0.0)
SOURCE_AIR_RADIUS_CM = 0.1
AIR_DENSITY_G_CM3 = 0.00001205


def box(lower, upper, **kwargs):
    """Create an axis-aligned CSG box from lower and upper XYZ coordinates."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(lower >= upper)):
        raise ValueError('Box bounds must be finite ordered XYZ coordinates')
    return openmc.model.RectangularParallelepiped(
        lower[0], upper[0], lower[1], upper[1], lower[2], upper[2], **kwargs)


def build_model(mesh_path=None, *, particles=1_000_000, batches=100, seed=10,
                container_lower=BOX_CONTAINER_LOWER_CM, container_upper=BOX_CONTAINER_UPPER_CM):
    """Place the unchanged sphere mesh in a translated box, beside a water box.

    Container bounds are in mesh-local coordinates. Other CSG bounds and the
    source location are in global coordinates. The source sphere is real air.
    """
    center = np.asarray(IRON_CENTER_CM)
    lower, upper = np.asarray(container_lower) + center, np.asarray(container_upper) + center
    source = np.asarray(SOURCE_CENTER_CM)
    if np.all(np.minimum(upper, WATER_UPPER_CM) > np.maximum(lower, WATER_LOWER_CM)):
        raise ValueError('Mesh container overlaps the water box')
    for lo, hi in [(lower, upper), (WATER_LOWER_CM, WATER_UPPER_CM),
                   (source - SOURCE_AIR_RADIUS_CM, source + SOURCE_AIR_RADIUS_CM)]:
        if np.any(np.asarray(lo) <= WORLD_LOWER_CM) or np.any(np.asarray(hi) >= WORLD_UPPER_CM):
            raise ValueError('Interior regions must fit strictly inside the vacuum box')

    iron = openmc.Material(material_id=1, name='iron')
    iron.add_element('Fe', 1.0)
    iron.set_density('g/cm3', 7.87)
    water = openmc.Material(material_id=2, name='water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.set_density('g/cm3', 1.0)
    air = openmc.Material(material_id=3, name='air')
    # Dry-air elemental mass fractions, including trace carbon.
    for element, fraction in [('N', .755267), ('O', .231781), ('Ar', .012827), ('C', .000125)]:
        air.add_element(element, fraction, percent_type='wo')
    air.set_density('g/cm3', AIR_DENSITY_G_CM3)

    mesh = openmc.UnstructuredMesh(filename=Path(mesh_path or INPUT_MESH).resolve(),
                                  library='moab', mesh_id=1)
    mesh.interface = 'xdg'
    universe = openmc.XDGUniverse(mesh)
    openmc.reserve_ids([1], openmc.Surface)
    openmc.reserve_ids([1], openmc.Cell)
    container = box(lower, upper)
    water_box = box(WATER_LOWER_CM, WATER_UPPER_CM)
    air_sphere = openmc.Sphere(x0=source[0], y0=source[1], z0=source[2], r=SOURCE_AIR_RADIUS_CM)
    outer = box(WORLD_LOWER_CM, WORLD_UPPER_CM, boundary_type='vacuum')
    mesh_cell = openmc.Cell(cell_id=20, name='dagmc_box', fill=universe,
                           region=-container & +air_sphere)
    mesh_cell.translation = IRON_CENTER_CM
    water_cell = openmc.Cell(cell_id=21, name='water_box', fill=water, region=-water_box)
    air_cell = openmc.Cell(cell_id=23, name='source_air', fill=air, region=-air_sphere)
    void_cell = openmc.Cell(cell_id=22, name='void', region=-outer & +container & +water_box & +air_sphere)
    geometry = openmc.Geometry([mesh_cell, water_cell, air_cell, void_cell])
    redundant_surfaces = geometry.remove_redundant_surfaces()

    neutron_source = openmc.IndependentSource()
    neutron_source.space = openmc.stats.Point(SOURCE_CENTER_CM)
    neutron_source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    neutron_source.particle = 'neutron'
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.source = neutron_source
    settings.batches, settings.particles, settings.seed = batches, particles, seed
    tally = openmc.Tally(tally_id=1, name='water_flux')
    tally.filters = [openmc.CellFilter([water_cell], filter_id=1)]
    tally.scores = ['flux']
    # tally.estimator = 'collision'

    # create a surface current tally on the negative-x face of the water box,
    # which is the face closest to the source
    box_x_min = water_box.xmin
    if box_x_min.id in redundant_surfaces:
        box_x_min = redundant_surfaces[box_x_min.id]

    # tally = openmc.Tally(tally_id=1, name='xmin_surface_current')
    # surf_filter = openmc.SurfaceFilter(box_x_min)
    # cell_from_filter = openmc.CellFromFilter(void_cell)
    # tally.filters = [surf_filter, cell_from_filter]
    # tally.scores = ['current']

    tallies = openmc.Tallies([tally])

    model = openmc.Model(geometry=geometry, materials=openmc.Materials([iron, water, air]),
                         settings=settings, tallies=tallies)
    model.geometry_derivatives = openmc.GeometricDerivatives([
        openmc.GeometricDerivative(tally, {1: ['all']}, id=1)])
    return model
