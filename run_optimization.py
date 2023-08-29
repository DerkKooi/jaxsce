"""Run optimization and compute integrals."""
import argparse
import json
import os
from timeit import default_timer as timer

# Numpy and jax for numerics
import jax.numpy as jnp
from jax.config import config

from jaxsce.constants import A
from jaxsce.coordinates_3d import get_coordinate_system
from jaxsce.densities.bohr_atom import BohrAtom
from jaxsce.densities.pyscf import CCSDDensity, HFDensity
from jaxsce.densities.sqrt_r import SqrtR
from jaxsce.integrate import VeeIntegration
from jaxsce.optimize import AngularOptimization, AngularOptimizationResult, TwoElectron

config.update("jax_enable_x64", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--density",
        type=str,
        default="sqrt_r",
        help="Which electron density to pick (sqrt_r, bohr_atom, hartree-fock, ccsd))",
    )
    parser.add_argument("--Nel", type=int, default=10, help="Number of electrons")
    parser.add_argument(
        "--atom", type=str, default="Ne", help="Atom to use for the HF density, e.g. 'Ne'"
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="cc-pVDZ",
        help="Basis to use for the HF density, e.g. 'cc-pVDZ'",
    )
    parser.add_argument("--charge", type=int, default=0, help="Charge of the HF density")
    parser.add_argument("--spin", type=int, default=0, help="Spin of the HF density")
    parser.add_argument(
        "--cumulant-threshold",
        type=float,
        default=1e-8,
        help="""Final point for the cumulant pre-inversion,
                Ne(r) = Nel - threshold, only relevant for Bohr atom and HF""",
    )
    parser.add_argument(
        "--N-points-Ne",
        type=int,
        default=10000,
        help="""Number of points to use for the pre-inversion of Ne(r),
        only relevant for Bohr atom and HF""",
    )
    parser.add_argument(
        "--invNe-tol",
        type=float,
        default=1e-10,
        help="Numerical tolerance for the inversion of Ne(r), only relevant for Bohr atom and HF",
    )
    parser.add_argument(
        "--coordinate-system",
        type=str,
        default="reduced",
        help="""Which coordinate system to use, 
                `allangles` full spherical coordinates is , 
                `reduced` means fix first electron z-axis, second electron in xz-plane, 
                `double reduced` means first electron in origin, second electron z-axis, 
                third electron in xz-plane""",
    )
    parser.add_argument(
        "--coordinate-system-last",
        type=str,
        default="double_reduced",
        help="Which coordinate system to use for the last point, see option `coordinate-system`",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="r",
        help="Which grid to use for integration, either equidistant in `r` or equidistant in `Ne`",
    )
    parser.add_argument(
        "--N-grid",
        type=int,
        default=1025,
        help="""Number of grid points for integration, 
                use 2^n + 1 for using Richardson extrapolation""",
    )
    parser.add_argument(
        "--N-random",
        type=int,
        default=10000,
        help="""Number of optimization attempts with random angles for every starting point, 
                provides starting point for sweeps if strategy is `sweeps`""",
    )
    parser.add_argument(
        "--N-random-last",
        type=int,
        default=30000,
        help="Number of attempts with random angles to minimize the last point",
    )
    parser.add_argument(
        "--nstart",
        type=int,
        nargs="+",
        default=[],
        help="""Which starting points to use for starting sweeps, 
                if empty then use first point from the border (left and right) 
                and right in the middle""",
    )
    parser.add_argument(
        "--N-select",
        type=int,
        default=30,
        help="Number of attempts to minimize the last point",
    )
    parser.add_argument(
        "--N-sweeps",
        type=int,
        default=0,
        help="Number of sweeps to make in the optimization, if zero then sweep until convergence",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sweeps",
        help="""Which strategy to use for the optimization, 
                default `sweeps`, other option is `random` """,
    )
    parser.add_argument(
        "--min-tol",
        type=float,
        default=1e-12,
        help="Tolerance for the minimization of Vee with respect to the angles",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Seed for the random number generator for generating angles",
    )
    parser.add_argument(
        "--equal-tol", type=float, default=1e-8, help="Tolerance for equality of two sets of angles"
    )
    parser.add_argument(
        "--unroll-inner-scan", type=int, default=1, help="Unroll inner scan (experimental)"
    )
    parser.add_argument(
        "--unroll-outer-scan", type=int, default=1, help="Unroll outer scan (experimental)"
    )
    parser.add_argument(
        "--check-serialization",
        type=bool,
        default=False,
        help="Check if the serialization worked as expected",
    )
    args = parser.parse_args()

    if args.nstart == []:
        # Start at the first point from the border and right in the middle.
        nstart = [1, args.N_grid // 2, args.N_grid - 2]
    else:
        nstart = args.nstart

    start_time = timer()

    # Make data directory
    if args.density in ("hartree-fock", "ccsd"):
        atom = args.atom[:2].strip()
        datadir = f"data/{atom}/{args.basis}"
        chkfile = "chkfile.chk"
    else:
        datadir = f"data/{args.density}/{args.Nel}"
    os.makedirs(datadir, exist_ok=True)

    # Build density
    if args.density == "sqrt_r":
        density = SqrtR(Nel=args.Nel)
    elif args.density == "bohr_atom":
        density = BohrAtom(
            args.Nel,
            cumulant_threshold=args.cumulant_threshold,
            N_points_Ne=args.N_points_Ne,
            invNe_tol=args.invNe_tol,
        )
    elif args.density == "hartree-fock":
        density = HFDensity(
            chkfile_name=chkfile,
            chkfile_dir=datadir,
            atom=atom,
            charge=args.charge,
            spin=args.spin,
            N_points_Ne=args.N_points_Ne,
            basis=args.basis,
            cumulant_threshold=args.cumulant_threshold,
            invNe_tol=args.invNe_tol,
        )
    elif args.density == "ccsd":
        dm_file = os.path.join(datadir, "dm_ccsd.npy")
        density = CCSDDensity(
            dm_file_name="dm_ccsd.npy",
            dm_file_dir=datadir,
            chkfile_dir=datadir,
            atom=atom,
            charge=args.charge,
            spin=args.spin,
            N_points_Ne=args.N_points_Ne,
            chkfile_name=chkfile,
            basis=args.basis,
            cumulant_threshold=args.cumulant_threshold,
            invNe_tol=args.invNe_tol,
        )
    else:
        raise ValueError(f"Unknown density: {args.density}")
    density_time = timer()

    if density.Nel == 2:
        coordinates_time = timer()
        opt = TwoElectron(N_grid=args.N_grid, grid=args.grid)
    else:
        # Build coordinates
        coordinates = get_coordinate_system(args.coordinate_system, args.seed)
        coordinates_last = get_coordinate_system(args.coordinate_system_last, args.seed)
        coordinates_time = timer()

        # Setup optimization
        opt = AngularOptimization(
            coordinates=coordinates,
            coordinates_last=coordinates_last,
            N_grid=args.N_grid,
            N_random=args.N_random,
            N_random_last=args.N_random_last,
            N_select=args.N_select,
            N_sweeps=args.N_sweeps,
            strategy=args.strategy,
            equal_tol=args.equal_tol,
            unroll_inner_scan=args.unroll_inner_scan,
            unroll_outer_scan=args.unroll_outer_scan,
        )
    opt_time = timer()

    # Run the optimization
    opt_result = opt.run(density, nstart)
    opt_run_time = timer()

    # Save the results
    if args.density == "ccsd":
        filename = f"{args.N_grid}_ccsd"
    else:
        filename = f"{args.N_grid}"
    opt_result.save(filename, datadir)
    save_time = timer()

    # Compute integrals
    ints = VeeIntegration(opt_result)
    ints.interpolate_all(4)
    integrals, extrapolated_integrals = ints.integrate(4)
    ints_time = timer()

    Winf = extrapolated_integrals["bpoly_2"][1] - density.U
    Lambda = -Winf / density.LDA_int
    B = (Winf - A * density.LDA_int) / density.GEA_int

    integral_dict = {
        "integrals": integrals,
        "extrapolated_integrals": extrapolated_integrals,
        "U": density.U,
        "LDA_int": density.LDA_int,
        "GEA_int": density.GEA_int,
        "Winf": Winf,
        "Lambda": Lambda,
        "B": B,
    }

    # Save integrals
    with open(f"{datadir}/{filename}_integrals.json", "w", encoding="utf-8") as f:
        json.dump(integral_dict, f)

    print(f"Time for density: {density_time - start_time}")
    print(f"Time for coordinates: {coordinates_time - density_time}")
    print(f"Time for optimization setup: {opt_time - coordinates_time}")
    print(f"Time for optimization run: {opt_run_time - opt_time}")
    print(f"Time for saving: {save_time - opt_run_time}")
    print(f"Time for integrals: {ints_time - save_time}")
    print(f"Total time: {ints_time - start_time}")

    if args.check_serialization:
        print("Check results class")

        opt_load = AngularOptimizationResult.load(filename, datadir)

        for key, val in opt_load.to_dict().items():
            if isinstance(val, jnp.ndarray):
                matches = jnp.allclose(val, getattr(opt_load, key), equal_nan=True)
                print(key, matches)
                if not matches:
                    print("Values for original", val)
                    print("Values for loaded", getattr(opt_load, key))

        print("Check params")
        for fieldname in opt_result.opt.__dataclass_fields__:
            if isinstance(getattr(opt_result.opt, fieldname), (int, float, str)):
                print(
                    fieldname,
                    getattr(opt_result.opt, fieldname) == getattr(opt_load.opt, fieldname),
                )

        print("Check density")
        print("name", opt_result.density.name == opt_load.density.name)
        print("Nel", opt_result.density.Nel == opt_load.density.Nel)

        print("Check coordinates")
        if density.Nel > 2:
            print("name", opt_result.opt.coordinates.name == opt_load.opt.coordinates.name)
            print("seed", opt_result.opt.coordinates.seed == opt_load.opt.coordinates.seed)
