from pathlib import Path

import openmc

IRON_CENTER_CM = (10.0, 0.0, 0.0)
SOURCE_CENTER_CM = (0.0, 0.0, 0.0)


def build_model(mesh_path=None, *, particles=1_000_000, batches=100, seed=10,
                container_radius=13.0):
    """Build separate iron mesh and water spheres inside a vacuum box."""
    iron = openmc.Material(material_id=1, name='iron')
    iron.add_element('Fe', 1.0)
    iron.set_density('g/cm3', 7.87)

    water = openmc.Material(material_id=2, name='water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.set_density('g/cm3', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')

    xdg_mesh = openmc.UnstructuredMesh(
        filename=(Path(__file__).resolve().parents[1] / 'sphere.h5m'
                  if mesh_path is None else Path(mesh_path).resolve()),
        library='moab',
        mesh_id=1,
    )
    xdg_mesh.interface = 'xdg'

    dagmc_universe = openmc.XDGUniverse(xdg_mesh)
    openmc.reserve_ids([1], openmc.Surface)
    openmc.reserve_ids([1], openmc.Cell)
    # Translate the radius-5 cm mesh to the reference iron sphere's center.
    # Its implicit complement fills the enclosing CSG sphere with void.
    iron_center = IRON_CENTER_CM
    inner_sphere = openmc.Sphere(
        surface_id=20, x0=iron_center[0], y0=iron_center[1],
        z0=iron_center[2], r=container_radius,
    )
    dagmc_cell = openmc.Cell(
        cell_id=20, name='dagmc_sphere',
        fill=dagmc_universe, region=-inner_sphere,
    )
    dagmc_cell.translation = iron_center

    water_sphere = openmc.Sphere(
        surface_id=21, x0=25.0, y0=0.0, z0=0.0, r=5.0,
    )
    water_cell = openmc.Cell(
        cell_id=21, name='water_sphere', fill=water, region=-water_sphere,
    )

    extent = water_sphere.x0 + water_sphere.r + 5.0
    outer_box = openmc.model.RectangularParallelepiped(
        -extent, extent, -extent, extent, -extent, extent,
        boundary_type='vacuum',
    )
    void_cell = openmc.Cell(
        cell_id=22, name='void',
        region=-outer_box & +inner_sphere & +water_sphere,
    )
    geometry = openmc.Geometry([dagmc_cell, water_cell, void_cell])

    # Match the reference model's isotropic 14.1 MeV neutron point source.
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point(SOURCE_CENTER_CM)
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    source.particle = 'neutron'

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.source = source
    settings.batches = batches
    settings.particles = particles
    settings.seed = seed

    water_filter = openmc.CellFilter([water_cell], filter_id=1)
    collision_filter = openmc.CollisionFilter([1])
    flux_tally = openmc.Tally(tally_id=1, name='water_flux')
    flux_tally.filters = [water_filter, collision_filter]
    flux_tally.scores = ['flux']
    # flux_tally.estimator = 'collision'

    surface_parameters = {1: ['all']}
    geom_deriv = openmc.GeometricDerivative(flux_tally, surface_parameters, id=1)

    model = openmc.Model(
        geometry=geometry,
        materials=openmc.Materials([iron, water]),
        settings=settings,
        tallies=openmc.Tallies([flux_tally]),
    )
    model.geometry_derivatives = openmc.GeometricDerivatives([geom_deriv])
    return model
