import openmc

from config import OPTIMIZATION_PARAMETER_NAMES

def build_model(parameter_names=None):
    """Build a fresh model with stable IDs for simulation workflows."""
    # ---------------------------------------------------------------------------
    # Materials
    # ---------------------------------------------------------------------------
    iron = openmc.Material(material_id=1, name='iron')
    iron.add_element('Fe', 1.0)
    iron.set_density('g/cm3', 7.87)

    water = openmc.Material(material_id=2, name='water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.set_density('g/cm3', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')

    materials = openmc.Materials([iron, water])

    # ---------------------------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------------------------
    # Iron sphere at the origin
    iron_sphere = openmc.Sphere(surface_id=1, x0=4, y0=5, z0=0, r=2)
    iron_cell = openmc.Cell(cell_id=1, name='iron_sphere', fill=iron, region=-iron_sphere)

    # Water sphere centered at (20, 0, 0)
    water_sphere = openmc.Sphere(surface_id=2, x0=25, y0=0, z0=0, r=5)
    water_cell = openmc.Cell(cell_id=2, name='water_sphere', fill=water, region=-water_sphere)

    # Large void sphere enclosing both cells with a vacuum boundary condition
    outer_sphere = openmc.Sphere(surface_id=3, x0=0, y0=0, z0=0, r=1000, boundary_type='vacuum')
    void_region = -outer_sphere & +iron_sphere & +water_sphere
    void_cell = openmc.Cell(cell_id=3, name='void', fill=None, region=void_region)

    geometry = openmc.Geometry([iron_cell, water_cell, void_cell])

    # ---------------------------------------------------------------------------
    # Source
    # ---------------------------------------------------------------------------
    # 14.1 MeV monoenergetic neutron point source at the origin, with an angular
    # distribution uniform over the positive-x hemisphere of the unit sphere.
    angle = openmc.stats.PolarAzimuthal(
        mu=openmc.stats.Uniform(0.0, 1.0),
        phi=openmc.stats.Uniform(0.0, 2 * 3.141592653589793),
        reference_uvw=(1.0, 0.0, 0.0),
        reference_vwu=(0.0, 1.0, 0.0),
    )

    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0, 0, 0))
    source.angle = angle
    source.energy = openmc.stats.Discrete([14.1e6], [1.0])
    source.particle = 'neutron'

    # ---------------------------------------------------------------------------
    # Settings
    # ---------------------------------------------------------------------------
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.source = source
    settings.batches = 100
    settings.particles = 1000000

    # ---------------------------------------------------------------------------
    # Tallies
    # ---------------------------------------------------------------------------
    water_filter = openmc.CellFilter([water_cell], filter_id=1)
    flux_tally = openmc.Tally(tally_id=1, name='water_flux')
    flux_tally.filters = [water_filter]
    flux_tally.scores = ['flux']
    flux_tally.estimator = 'collision'

    tallies = openmc.Tallies([flux_tally])


    geom_derivative = openmc.GeometricDerivative(flux_tally, iron_sphere, id=1)
    geom_derivative.surface_parameters = list(
        OPTIMIZATION_PARAMETER_NAMES if parameter_names is None else parameter_names
    )

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------
    model = openmc.Model(geometry=geometry, materials=materials,
                          settings=settings, tallies=tallies)

    model.geometry_derivatives = [geom_derivative]

    return model


def main(output_path="model.xml", parameter_names=None):
    """Generate and export the model used by optimization and sweeps."""
    model = build_model(parameter_names)
    model.export_to_model_xml(output_path)
    return model


if __name__ == "__main__":
    main()
