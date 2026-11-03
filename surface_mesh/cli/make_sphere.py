"""Generate a starting sphere mesh using optional Cubit bindings."""

import argparse
from ..core.sphere import generate_sphere


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a sphere triangle mesh using Cubit."
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Sphere radius (default: 5.0)",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="Sphere center coordinates (default: 0 0 0)",
    )
    parser.add_argument(
        "--anisotropic-ratio",
        type=float,
        default=100.0,
        help="Coarse trimesher anisotropic ratio (default: 100)",
    )
    parser.add_argument(
        "--deviation-angle",
        type=float,
        default=5.0,
        help="Coarse trimesher deviation angle in degrees (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sphere.h5m",
        help="Output filename (.stl or .h5m; default: sphere.h5m)",
    )

    args = parser.parse_args(argv)

    generate_sphere(
        radius=args.radius,
        origin=tuple(args.origin),
        anisotropic_ratio=args.anisotropic_ratio,
        deviation_angle=args.deviation_angle,
        output=args.output,
    )


if __name__ == '__main__':
    raise SystemExit(main())
