"""Generate a triangulated sphere using the Cubit Python API."""

from pathlib import Path


def generate_sphere(
    radius: float = 5.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    anisotropic_ratio: float = 100.0,
    deviation_angle: float = 5.0,
    output: str = "sphere.h5m",
) -> None:
    """Create, mesh, and export a sphere surface.

    Parameters
    ----------
    radius
        Sphere radius.
    origin
        Sphere center coordinates (x, y, z).
    anisotropic_ratio
        Anisotropic ratio used by Cubit's coarse trimesher.
    deviation_angle
        Coarse trimesher deviation angle in degrees.
    output
        Output filename (.stl or .h5m).
    """
    import cubit

    if radius <= 0:
        raise ValueError("radius must be positive")
    if anisotropic_ratio <= 0:
        raise ValueError("anisotropic_ratio must be positive")
    if not 0 < deviation_angle <= 180:
        raise ValueError("deviation_angle must be in (0, 180)")
    if len(origin) != 3:
        raise ValueError("origin must contain three coordinates")

    output = Path(output).expanduser().resolve()
    extension = output.suffix.lower()

    if extension not in (".stl", ".h5m"):
        raise ValueError(
            f"Unsupported output format: {extension}. "
            "Supported formats are .stl and .h5m."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    x, y, z = origin

    # Initialize Cubit if needed.
    cubit.init(["cubit", "-nojournal"])

    def command(text: str) -> None:
        if not cubit.cmd(text):
            raise RuntimeError(f"Cubit command failed: {text}")

    # Reset the model.
    command("reset")

    # Create sphere and translate to requested center.
    command(f"create sphere radius {radius:.16g}")
    command(f"volume all move {x:.16g} {y:.16g} {z:.16g}")

    # Configure coarse TriMesh settings.
    command(
        f"set trimesher coarse on "
        f"ratio {anisotropic_ratio:.16g} "
        f"angle {deviation_angle:.16g}"
    )

    # Assign triangular surface meshing scheme.
    command("surface all scheme trimesh")

    # Generate triangle surface mesh.
    command("mesh surface all")

    # Export according to output file extension.
    if extension == ".stl":
        command(f'export stl "{output}" mesh overwrite')
    elif extension == ".h5m":
        command(f'export cf_dagmc "{output}" overwrite')

    print(f"Sphere mesh written to: {output}")
