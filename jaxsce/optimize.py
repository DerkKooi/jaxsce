"""Contains the optimizers for the JAXSCE package."""

import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from dataclasses_json import config, dataclass_json
from jax import jacrev, jit, lax, vmap
from jaxopt import BFGS, LBFGS, GradientDescent, NonlinearCG, ScipyMinimize
from jaxopt.base import IterativeSolver

from .coordinates_3d import SphericalCoordinateSystem, get_coordinate_system
from .densities import get_density
from .densities.classes import Density

jit_methods = [
    "_run_last",
    "_run_grid_random_min",
    "_run_grid_random",
    "_run_angles_only",
    "_step",
    "_sweep_while",
    "_sweep_scan",
    "dVee",
    "d2Vee",
    "d2Vee_slow",
    "lowest_eigvalsh_hessian",
    "lowest_eigvalsh_hessian_last",
    "sort_angles_Vee",
]


@jit
def improvement_branch(
    k: int,
    angles: jnp.ndarray,
    Vee: jnp.ndarray,
    new_angles: jnp.ndarray,
    new_Vee: jnp.ndarray,
    _,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Branch for when the optimization improves the Vee value.

    Arguments
    ---------
    k: int
        Index of the angle that was changed.
    angles: jnp.ndarray
        Current angles.
    Vee: jnp.ndarray
        Current Vee values.
    new_angles: jnp.ndarray
        New angles.
    new_Vee: jnp.ndarray
        New Vee values.

    Returns
    -------
    angles: jnp.ndarray
        Updated angles.
    Vee: jnp.ndarray
        Updated Vee values.
    fail_count: int
        Reset fail_count to 0.
    """
    return (angles.at[k].set(new_angles), Vee.at[k].set(new_Vee), 0)


@jit
def no_improvement_branch(
    _1,
    angles: jnp.ndarray,
    Vee: jnp.ndarray,
    _2,
    _3,
    fail_count: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Branch for when the optimization does not improve the Vee value.

    Arguments
    ---------
    angles: jnp.ndarray
        Current angles.
    Vee: jnp.ndarray
        Current Vee values.
    fail_count: int
        Current fail_count.

    Returns
    -------
    angles: jnp.ndarray
        Unchanged angles.
    Vee: jnp.ndarray
        Unchanged Vee values.
    fail_count: int
        Updated fail_count + 1.
    """
    return (angles, Vee, fail_count + 1)


@jit
def forward_branch(f: jnp.ndarray) -> jnp.ndarray:
    """
    Branch for when moving forward in the sweep optimization.

    Arguments
    ---------
    f: jnp.ndarray
        Co-motion functions.
    Returns
    -------
    f: jnp.ndarray
        Unchanged co-motion functions.
    """
    return f


@jit
def reverse_branch(f: jnp.ndarray) -> jnp.ndarray:
    """
    Branch for when moving backward in the sweep optimization.

    Arguments
    ---------
    f: jnp.ndarray
        Co-motion functions.

    Returns
    -------
    f: jnp.ndarray
        Flipped co-motion functions.
    """
    return jnp.flip(f, axis=0)


@jit
def while_condition(N_grid: int, carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool, int]) -> bool:
    """
    Condition for the while loop in the sweep optimization.

    Arguments
    ---------
    N_grid: int
        Number of grid points.
    carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool, int]
        Carry tuple.

    Returns
    -------
    return: bool
        True if fail_count < N_grid, the while loop should continue.
    """
    return carry[2] < N_grid


@jit
def return_forward_branch(
    angles: jnp.ndarray, Vee: jnp.ndarray, count: int
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Return the angles and Vee values if the last sweep was reverse.

    Arguments
    ---------
    angles: jnp.ndarray
        Current angles.
    Vee: jnp.ndarray
        Current Vee values.
    count: int
        Current fail_count (scan) or sweep_count (while).

    Returns
    -------
    angles: jnp.ndarray
        Flipped angles.
    Vee: jnp.ndarray
        Flipped Vee values.
    count: int
        Unchanged fail_count (scan) or sweep_count (while).
    """
    return jnp.flip(angles, axis=0), jnp.flip(Vee, axis=0), count


@jit
def return_reverse_branch(
    angles: jnp.ndarray, Vee: jnp.ndarray, count: int
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Return the angles and Vee values if the last sweep was foward.

    Arguments
    ---------
    angles: jnp.ndarray
        Current angles.
    Vee: jnp.ndarray
        Current Vee values.
    count: int
        Current fail_count (scan) or sweep_count (while).

    Returns
    -------
    angles: jnp.ndarray
        Unchanged angles.
    Vee: jnp.ndarray
        Unchanged Vee values.
    count: int
        Unchanged fail_count (scan) or sweep_count (while).
    """
    return angles, Vee, count


@dataclass_json
@dataclass(eq=False)
class AngularOptimization:
    """
    Optimize the angles of the electrons for all the N_grid points,
    where the radii are given by co-motion functions.

    Attributes
    ----------
    coordinates : SphericalCoordinateSystem
        The coordinate system to use.
    N_grid : int
        The number of grid points to use.
    N_random : int
        The number of optimized random starting points to use.
    N_random_last : int
        The number of optimized random starting points to use for the last coordinate.
    coordinates_last : Optional[SphericalCoordinateSystem]
        The coordinate system to use for the last coordinate.
    N_select : Optional[int]
        The number of optimized starting points to select for every starting point.
    N_sweeps : int
        The number of sweeps to use for the optimization. If zero, then sweep to convergence
    grid : str
        The grid to use, either equidistant on "r" or "Ne". Default is "r".
    strategy : str
        The strategy to use, either "all random" or "random grid". Default is "all random".
    method : str
        The method to use to minimize the electron repulsion. Default is "BFGS".
    maxiter : int
        The maximum number of iterations for the solver. Default is 5000.
    tol : float
        The tolerance for the solver. Default is 1e-12.
    implicit_diff : bool
        Whether to use implicit differentiation in the solver. Default is True.
    equal_tol : Optional[float]
        The tolerance to use for the equality of two sets of angles
        in terms of 1/N sum_i=1^N |r_i - r_i'|. Default is 1e-8.
    solver : jaxopt.IterativeSolver
        The jaxopt solver.
    solver_last : jaxopt.IterativeSolver
        The jaxopt solver to use for the last coordinate.
    """

    coordinates: SphericalCoordinateSystem = field(
        metadata=config(
            encoder=lambda x: x.encode(),
            decoder=lambda y: get_coordinate_system(y["name"], y["seed"]),
        )
    )
    N_grid: int
    N_random: int
    N_random_last: int
    coordinates_last: SphericalCoordinateSystem = field(
        default=None,
        metadata=config(
            encoder=lambda x: x.encode(),
            decoder=lambda y: get_coordinate_system(y["name"], y["seed"]),
        ),
    )
    N_select: Optional[int] = None
    N_sweeps: int = 0
    grid: str = "r"
    strategy: int = "sweeps"
    method: str = "BFGS"
    maxiter: int = 5000
    tol: float = 1e-12
    implicit_diff: bool = True
    equal_tol: Optional[float] = 1e-8
    unroll_inner_scan: int = 1
    unroll_outer_scan: int = 1
    methods_jaxopt: ClassVar[Dict[str, IterativeSolver]] = {
        "BFGS": BFGS,
        "GradientDescent": GradientDescent,
        "LBFGS": LBFGS,
        "NonlinearCG": NonlinearCG,
    }  # Dict of all the jaxopt solvers
    methods_scipy: ClassVar[Tuple[str, ...]] = (
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS_scipy",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    )  # Tuple of all the scipy solvers

    def __post_init__(self):
        if self.strategy == "all random":
            # When using all random these options are ignored
            self.equal_tol = None
            self.N_select = None

        if self.coordinates_last is None:
            # If no coordinate system is given for the last coordinate,
            # use the same as for the other coordinates
            self.coordinates_last = self.coordinates

        # Select the solver from jaxopt
        if self.method in self.methods_jaxopt:
            jax_kwargs = {
                "value_and_grad": True,
                "maxiter": self.maxiter,
                "tol": self.tol,
                "jit": True,
                "implicit_diff": self.implicit_diff,
            }

            self.solver = self.methods_jaxopt[self.method](
                self.coordinates.Vee_value_and_grad,
                **jax_kwargs,
            )
            self.solver_last = self.methods_jaxopt[self.method](
                self.coordinates_last.Vee_value_and_grad,
                **jax_kwargs,
            )
        elif self.method in self.methods_scipy:
            # Select the solver from scipy
            if self.method == "BFGS_scipy":
                # Avoid clash with jaxopt BFGS
                method_scipy = "BFGS"
            else:
                method_scipy = self.method
            self.solver = ScipyMinimize(
                fun=self.coordinates.Vee,
                jit=True,
                method=method_scipy,
                tol=self.tol,
                maxiter=self.maxiter,
            )
        else:
            raise NotImplementedError(f"Method {self.method} not implemented.")

        # Set up a function that only returns the angles and the value of the function
        def _run_Vee_and_angles(start_angles: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
            angles, state = self.solver.run(start_angles, f)
            return angles, state.value

        # Same for the last coordinate
        def _run_Vee_and_angles_last(start_angles: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
            angles, state = self.solver_last.run(start_angles, f)
            return angles, state.value

        # Jit and vmap the _run_Vee_and_angles function
        self._run = jit(_run_Vee_and_angles)
        self._run_vmap = jit(vmap(_run_Vee_and_angles, in_axes=(0, None)))
        self._run_grid_vmap = jit(
            vmap(vmap(_run_Vee_and_angles, in_axes=(0, None)), in_axes=(0, 0))
        )
        self._run_last_vmap = jit(vmap(_run_Vee_and_angles_last, in_axes=(0, None)))

        # Jit all the functions listed in jit_methods, after appropriate values have been set
        for method in jit_methods:
            setattr(self, method, jit(getattr(self, method)))

        # Jit the sweep functions with n_start as static argument
        self._first_sweep = jit(self._first_sweep, static_argnums=(0,))
        self._first_sweep_vmap = jit(
            vmap(self._first_sweep, in_axes=(None, None, 0, 0), out_axes=(0, 0, 0)),
            static_argnums=0,
        )
        if self.N_sweeps == 0:
            self._run_sweeps = jit(self._run_sweeps_while)
            self._run_sweeps_vmap = jit(
                vmap(self._run_sweeps_while, in_axes=(None, 0, 0), out_axes=(1, 1, 0))
            )
        else:
            self._run_sweeps = jit(self._run_sweeps_scan)
            self._run_sweeps_vmap = jit(
                vmap(self._run_sweeps_scan, in_axes=(None, 0, 0), out_axes=(1, 1, 0)),
            )

        # Jit and vmap the functions that returns the N_select unique angles
        self.N_unique_angles_vmap = jit(vmap(self.N_unique_angles, in_axes=(0, 0)))
        self.return_unique_angles_vmap = jit(vmap(self.return_unique_angles, in_axes=(0, 0, 0)))

        # Jit and vmap the derivative functions
        self.dVee_vmap = jit(vmap(self.dVee, in_axes=(0, 0, 0)))
        self.d2Vee_vmap = jit(vmap(self.d2Vee, in_axes=(0, 0, 0, 0)))

    def _step(
        self, f: jnp.ndarray, inner_carry: Tuple[jnp.ndarray, jnp.ndarray, int], k: int
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, int], None]:
        """
        Perform one step of the sweep optimization.

        Parameters
        ----------
        f: jnp.ndarray of shape (N_grid_end, Nel)
            The values of the co-motion functions.
        inner_carry: Tuple[jnp.ndarray, jnp.ndarray, int]
            The inner carry (angles, Vee, fail_count) of the sweep optimization.
        k: int
            The index of the gridpoint to optimize.

        Returns
        -------
        new_carry: Tuple[jnp.ndarray, jnp.ndarray, int]
            The new inner carry (angles, Vee, fail_count) of the sweep optimization.
        None
            Needs to be returned to satisfy the jax.lax.scan signature.
        """

        # Unpack carry
        angles, Vee, fail_count = inner_carry

        # Run the optimization for point k with the k-1 angles as guess
        new_angles, new_Vee = self._run(angles[k - 1], f[k])
        # Check for improvement, if yes: update angles and Vee, if no: increase fail_count

        new_carry = lax.cond(
            new_Vee < Vee[k],
            improvement_branch,
            no_improvement_branch,
            k,
            angles,
            Vee,
            new_angles,
            new_Vee,
            fail_count,
        )
        return new_carry, None

    def _first_sweep(
        self, n_start: int, f_flip: jnp.ndarray, angles: jnp.ndarray, Vee: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Perform a sweep optimization, backwards from n_start to the first grid point.

        Parameters
        ----------
        f_flip: jnp.ndarray (N_grid_end, N_el)
            The flipped (on the grid) co-motion functions.
        n_start: int
            The index of the first point to start the sweep from.
        angles: jnp.ndarray (N_grid_end, x)
            The angles to start the sweep from.
        Vee: jnp.ndarray (N_grid_end,)
            The energies to start the sweep from.

        Returns
        -------
        first_carry: Tuple[jnp.ndarray, jnp.ndarray, int]
            angles: jnp.ndarray (N_grid_end, x)
                The angles after the sweep. (Flipped)
            Vee: jnp.ndarray (N_grid_end,)
                The energies after the sweep. (Flipped)
            fail_count: int
                The number of times the optimization failed to improve the energy.
                This is zero anyway, since Vee is initialized as infinity.
        """
        # Extract the number of grid points
        N_grid = f_flip.shape[0]

        # We are going backwards, so we need to flip the angles and Vee
        first_carry = (
            jnp.flip(angles, axis=0),
            jnp.flip(Vee, axis=0),
            0,
        )

        # Carry out the grid scan from n_start-1 to 0
        first_carry, _ = lax.scan(
            partial(self._step, f_flip), first_carry, jnp.arange(N_grid - n_start, N_grid)
        )
        return first_carry

    def _sweep_while(
        self,
        f: jnp.ndarray,
        outer_carry: Tuple[jnp.ndarray, jnp.ndarray, bool, int],
    ) -> jnp.ndarray:
        """
        Perform a sweep optimization, either forward or backwards,
        depending on the value of forward.

        Parameters
        ----------
        f: jnp.ndarray (N_grid_end, N_el)
            The co-motion functions.
        outer_carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool, int]
            angles: jnp.ndarray (N_grid_end, x)
                The angles to start the sweep from.
            Vee: jnp.ndarray (N_grid_end,)
                The energies to start the sweep from.
            fail_count: int
                The number of times the optimization failed to improve the energy.
            forward: bool
                Whether to perform the sweep forward or backwards.
            sweep_count: int
                The number of sweeps performed.

        Returns
        -------
        outer_carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool, int]
            As above, but with the updated attributes.
        """

        # Unpack outer_carry
        angles, Vee, fail_count, forward, sweep_count = outer_carry
        N_grid = angles.shape[0]
        inner_carry = (
            jnp.flip(angles, axis=0),
            jnp.flip(Vee, axis=0),
            fail_count,
        )

        inner_carry, _ = lax.scan(
            partial(self._step, lax.cond(forward, forward_branch, reverse_branch, f)),
            inner_carry,
            jnp.arange(N_grid),
            unroll=self.unroll_inner_scan,
        )
        return (*inner_carry, jnp.logical_not(forward), sweep_count + 1)

    def _sweep_scan(
        self,
        f: jnp.ndarray,
        outer_carry: Tuple[jnp.ndarray, jnp.ndarray, bool, int],
        x: Optional[int],
    ) -> jnp.ndarray:
        """
        Perform a sweep optimization, either forward or backwards,
        depending on the value of forward.

        Parameters
        ----------
        f: jnp.ndarray (N_grid_end, N_el)
            The co-motion functions.
        f_flip: jnp.ndarray (N_grid_end, N_el)
            The flipped (on the grid) co-motion functions.
        outer_carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool]
            angles: jnp.ndarray (N_grid_end, x)
                The angles to start the sweep from.
            Vee: jnp.ndarray (N_grid_end,)
                The energies to start the sweep from.
            fail_count: int
                The number of times the optimization failed to improve the energy.
            forward: bool
                Whether to perform the sweep forward or backwards.
        x: Optional[int]
            A variable passed to indicate where we are in the sweep with scan, unused.

        Returns
        -------
        outer_carry: Tuple[jnp.ndarray, jnp.ndarray, int, bool]
            As above, but with the updated attributes.
        None
        """

        # Unpack outer_carry
        angles, Vee, fail_count, forward = outer_carry
        N_grid = angles.shape[0]
        inner_carry = (
            jnp.flip(angles, axis=0),
            jnp.flip(Vee, axis=0),
            fail_count,
        )

        inner_carry, _ = lax.scan(
            partial(self._step, lax.cond(forward, forward_branch, reverse_branch, f)),
            inner_carry,
            jnp.arange(N_grid),
            unroll=self.unroll_inner_scan,
        )
        return (*inner_carry, jnp.logical_not(forward)), x

    def _run_sweeps_while(
        self, f: jnp.ndarray, angles: jnp.ndarray, Vee: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Perform the sweep optimization by doing sweeps until Vee doesn't change anymore.

        Parameters
        ----------
        f: jnp.ndarray (N_grid_end, N_el)
            The co-motion functions.
        angles: jnp.ndarray (N_grid, x,)
            The angles to start the sweep from the first sweep.
        Vee: float
            The energy to start the sweep from the first sweep.

        Returns
        -------
        angles: jnp.ndarray (N_grid_end, x)
            The angles after the sweep.
        Vee: jnp.ndarray (N_grid_end,)
            The energies after the sweep.
        sweep_count: int
            The number of sweeps performed.
        """
        N_grid = f.shape[0]

        # Carry out the rest of the sweeps, until converged
        angles, Vee, _, direction, sweep_count = lax.while_loop(
            partial(while_condition, N_grid),
            partial(self._sweep_while, f),
            (angles, Vee, 0, True, 0),
        )

        # Flip the angles and Vee if we went backwards on the last step
        return lax.cond(
            direction, return_forward_branch, return_reverse_branch, angles, Vee, sweep_count
        )

    def _run_sweeps_scan(
        self, f: jnp.ndarray, angles: jnp.ndarray, Vee: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Perform the sweep optimization by doing N_sweeps sweeps.

        Parameters
        ----------
        f: jnp.ndarray (N_grid_end, N_el)
            The co-motion functions.
        angles: jnp.ndarray (N_grid, x,)
            The angles to start the sweep from the first sweep.
        Vee: float
            The energy to start the sweep from the first sweep.

        Returns
        -------
        angles: jnp.ndarray (N_grid_end, x)
            The angles after the sweep.
        Vee: jnp.ndarray (N_grid_end,)
            The energies after the sweep.
        sweep_count: int
            The number of sweeps performed.
        """

        # Carry out the rest of the sweeps, until converged
        carry, _ = lax.scan(
            partial(self._sweep_scan, f),
            (angles, Vee, 0, True),
            xs=None,
            length=self.N_sweeps,
            unroll=self.unroll_outer_scan,
        )

        angles, Vee, fail_count, direction = carry

        # Flip the angles and Vee if we went backwards on the last step
        return lax.cond(
            direction, return_forward_branch, return_reverse_branch, angles, Vee, fail_count
        )

    def run(
        self, density: Density, n_start: Optional[Union[jnp.ndarray, int, List[int]]] = None
    ) -> "AngularOptimizationResult":
        """
        Run the optimization.

        Parameters
        ----------
        n_start: Union[jnp.ndarray, int, List[int]]
            The index of the first point to start the sweep from.
            If an array or list, the optimization will be run for each value in the array or list.
            If an int, the optimization will be run for that value.

        Returns
        -------
        result: AngularOptimizationResult
            The result of the optimization.
        """

        if isinstance(n_start, int):
            n_start = jnp.array([n_start])
        elif isinstance(n_start, list):
            n_start = jnp.array(n_start)

        # Set up the grid, from the point where the density integrates to 1
        # to the point where the density integrates to 2.
        print("Setting up grid, calculating density and co-motion functions...")
        if self.grid == "r":
            r = jnp.linspace(density.a[1], density.a[2], self.N_grid)
            Ne = density.Ne(r)
            f = density.co_motion_functions(r)
        elif self.grid == "Ne":
            Ne = jnp.linspace(1, 2, self.N_grid)
            r = density.invNe(Ne)
            f = density.co_motion_functions_Ne(Ne)
        else:
            raise NotImplementedError(f"Grid {self.grid} not implemented.")

        # Calculate density, derivatives and coNe integral
        print("Calculating density, derivatives and coNe integral...")
        rho = density.rho(r)
        drho = density.rho_deriv(r)
        d2rho = density.rho_deriv2(r)
        coNe = density.coNe(density.invNe(jnp.arange(density.Nel)))

        if density.Nel % 2 == 0:
            print("Calculating last point separately...")
            # If there is an even number of electrons, then
            # at the last grid point one of the electrons is at infinity
            # therefor we treat it separately with this electron removed
            N_grid_end = self.N_grid - 1
            f_last = jnp.delete(f[-1], density.Nel - 2)
            angles_last, Vee_last = self._run_last(f_last)

            # Split the key to get new random numbers
            self.coordinates.split_key()

            # Check if the last point is a local minimum
            # because only electron is at zero, two angle eigenvalues are zero
            # therefore check if the lowest eigenvalue is larger than -1e-12
            local_minimum_last = self.lowest_eigvalsh_hessian_last(angles_last, f_last) > -1e-12
        else:
            # For an odd number of electrons, just run the whole grid in the same way
            N_grid_end = self.N_grid
            angles_last = None
            Vee_last = None
            local_minimum_last = None

        if self.strategy == "all random":
            print("Running random starting calculations for every point on the grid...")
            # Run random starting calculations for every point on the whole grid
            angles, Vee = self._run_grid_random_min(f[:N_grid_end])
        elif self.strategy == "sweeps":
            print("Running random starting calculations for starting points...")
            # Run random calculations on the starting points
            angles_start, Vee_start = self._run_grid_random(f[n_start])
            print("Sorting angles and energies...")
            # Sort the starting points by energy
            angles_start, Vee_start = self.sort_angles_Vee(angles_start, Vee_start)
            print("Selecting unique starting points for sweep...")
            # Select the N_select unique lowest energy starting points
            N_unique = self.N_unique_angles_vmap(angles_start, f[n_start])
            print(
                f"""Number of unique starting points: {N_unique},
                fraction: {N_unique/self.N_random}, N_select: {self.N_select}"""
            )

            angles_select, Vee_select = self.return_unique_angles_vmap(
                angles_start, Vee_start, f[n_start]
            )

            for i in range(n_start.shape[0]):
                if N_unique[i] < self.N_select:
                    N_repeats = int(jnp.ceil((self.N_select - N_unique[i]) / N_unique[i])) + 1
                    angles_select = angles_select.at[i].set(
                        jnp.repeat(angles_select[i, : N_unique[i]], N_repeats, axis=0)[
                            : self.N_select
                        ]
                    )

            print("Running first sweep...")
            # Run the first sweep, we need static compilation for every starting point
            # so we use a simple python loop
            angles: List[jnp.ndarray] = []
            Vee: List[jnp.ndarray] = []
            for i, n_start_i in enumerate(n_start):
                print(f"Running first sweep for starting point {n_start_i}")
                angles_fill = (
                    self.coordinates.empty_angles(f.shape[1], self.N_select, N_grid_end)
                    .at[:, n_start_i]
                    .set(angles_select[i])
                )
                Vee_fill = (
                    (jnp.ones((self.N_select, N_grid_end)) * jnp.inf)
                    .at[:, n_start_i]
                    .set(Vee_select[i])
                )
                angles_i, Vee_i, _ = self._first_sweep_vmap(
                    int(n_start_i), jnp.flip(f[:N_grid_end], axis=0), angles_fill, Vee_fill
                )
                angles.append(angles_i)
                Vee.append(Vee_i)

            # Concatenate the results from the different starting points
            angles = jnp.concatenate(angles, axis=0)
            Vee = jnp.concatenate(Vee, axis=0)

            # Sweep over all starting points
            print("Running sweeps...")
            angles, Vee, count = self._run_sweeps_vmap(f[:N_grid_end], angles, Vee)
            if self.N_sweeps == 0:
                print("Sweep counts:", count)
            else:
                print("fail counts:", count)
                print("fail counts below N_grid_end", jnp.where(count < N_grid_end))

            print("Minimizing over the different starting points...")
            # Find the minimum over starting points for every grid point
            min_idx = jnp.argmin(Vee.reshape(N_grid_end, -1), axis=1)

            # Find the starting point that gave the minimum for each grid point
            min_starting_point = min_idx // self.N_select

            # Find the index of the optimized random starting point for each grid point
            min_random = min_idx % self.N_select

            # Select the angles and energies for the minimum
            angles = angles.reshape(N_grid_end, -1, angles.shape[-1])[
                jnp.arange(N_grid_end), min_idx
            ]
            Vee = Vee.reshape(N_grid_end, -1)[jnp.arange(N_grid_end), min_idx]
        else:
            raise NotImplementedError(f"Strategy {self.strategy} not implemented.")

        # Split the key in self.coordinates for if it is used in the future
        self.coordinates.split_key()

        print("Checking if grid points are local minima...")
        # Check if the grid points are a local minimum
        local_minimum = self.lowest_eigvalsh_hessian(angles, f[:N_grid_end]) >= -1e-12

        # Add the last point if the number of electrons is even
        if density.Nel % 2 == 0:
            Vee = jnp.concatenate([Vee, jnp.array([Vee_last])], axis=0)

        # Calculate derivatives w.r.t. r for interpolation
        # Derivatives don't make sense for the last point
        print("Calculating derivatives of co-motion functions...")
        if self.grid == "r":
            df = density.co_motion_functions_deriv(r, f)
            d2f = density.co_motion_functions_deriv2(r, f, df)

        elif self.grid == "Ne":
            df = density.co_motion_functions_deriv_Ne(Ne, f)
            d2f = density.co_motion_functions_deriv2_Ne(Ne, f, df)
        else:
            raise NotImplementedError(f"Grid {self.grid} not implemented.")

        print("Calculating first derivative of Vee...")
        dVee = self.dVee_vmap(angles[:N_grid_end], f[:N_grid_end], df[:N_grid_end])
        print("Calculating second derivative of Vee...")
        d2Vee = self.d2Vee_vmap(
            angles[:N_grid_end], f[:N_grid_end], df[:N_grid_end], d2f[:N_grid_end]
        )
        print("End of optimization.")

        return AngularOptimizationResult(
            opt=self,
            density=density,
            r=r,
            rho=rho,
            drho=drho,
            d2rho=d2rho,
            Ne=Ne,
            f=f,
            df=df,
            d2f=d2f,
            coNe=coNe,
            angles=angles,
            angles_last=angles_last,
            Vee=Vee,
            dVee=dVee,
            d2Vee=d2Vee,
            local_minimum_last=local_minimum_last,
            local_minimum=local_minimum,
            min_starting_point=min_starting_point,
            min_random=min_random,
            n_start=n_start,
        )

    def _run_last(self, f_last: jnp.ndarray) -> jnp.ndarray:
        """
        Run the last point without diverging co-motion function.

        Parameters
        ----------
        f_last : jnp.ndarray
            The co-motion functions at the last point.

        Returns
        -------
        jnp.ndarray
            The optimized angles at the last point.
        """
        start_angles_last = self.coordinates_last.random_angles(f_last.shape[0], self.N_random_last)
        angles_last, Vee_last = self._run_last_vmap(start_angles_last, f_last)
        min_index_last = jnp.argmin(Vee_last)

        return angles_last[min_index_last], Vee_last[min_index_last]

    def _run_grid_random(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Optimize the angles at the grid points from random starting positions.

        Returns all.

        Parameters
        ----------
        f : jnp.ndarray (N_grid -1, Nel)
            The co-motion functions at the grid points.

        Returns
        -------
        angles : jnp.ndarray (N_grid - 1, N_random, x)
            The optimized angles at the grid points.
        Vee : jnp.ndarray (N_grid - 1, N_random)
            The optimized energies at the grid points.
        """

        # Run the first N_grid - 1 points
        Nel = f.shape[1]
        start_angles = self.coordinates.random_angles(Nel, f.shape[0], self.N_random)
        angles, Vee = self._run_grid_vmap(start_angles, f)
        return angles, Vee

    def _run_grid_random_min(self, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Optimize the angles at the grid points from random starting positions.

        Returns only the minimum.

        Parameters
        ----------
        f : jnp.ndarray (N_grid -1, Nel)
            The co-motion functions at the grid points.

        Returns
        -------
        angles : jnp.ndarray (N_grid - 1, x)
            The optimized angles at the grid points.
        Vee : jnp.ndarray (N_grid - 1)
            The optimized energies at the grid points.
        """

        # Run the first N_grid - 1 points
        angles, Vee = self._run_grid_random(f)
        min_indices = jnp.argmin(Vee, axis=1)

        # Take the minimum
        angles = jnp.take_along_axis(angles, min_indices[:, None, None], axis=1)[:, 0]
        Vee = jnp.take_along_axis(Vee, min_indices[:, None], axis=1)[:, 0]
        return angles, Vee

    def dVee(self, angles: jnp.ndarray, f: jnp.ndarray, df: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the derivative of Vee w.r.t. r or Ne.

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.
        df : jnp.ndarray (N_grid - 1, Nel)
            The derivative of the co-motion functions at the grid points.

        Returns
        -------
        dVee: jnp.ndarray (N_grid - 1)
            The derivative of Vee w.r.t. r or Ne.
        """
        dVee_df = self.coordinates.Vee_grad_radials(angles, f)
        return jnp.sum(df * dVee_df)

    def _run_angles_only(self, angles: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """
        Run the optimization for a given set of angles and co-motion functions.

        Return only the angles (for differentiation.)

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.

        Returns
        -------
        opt_angles : jnp.ndarray (N_grid - 1, x)
            The optimized angles at the grid points.
        """
        opt_angles, _ = self.solver.run(angles, f)
        return opt_angles

    def d2Vee_slow(
        self, angles: jnp.ndarray, f: jnp.ndarray, df: jnp.ndarray, d2f: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the second derivative of Vee w.r.t. r.

        This is the slow version, which is used for testing.

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.
        df : jnp.ndarray (N_grid - 1, Nel)
            The derivative of the co-motion functions at the grid points.
        d2f : jnp.ndarray (N_grid - 1, Nel)
            The second derivative of the co-motion functions at the grid points.

        Returns
        -------
        d2Vee: jnp.ndarray (N_grid - 1)
            The second derivative of Vee w.r.t. r.
        """
        dVee_df = self.coordinates.Vee_grad_radials(angles, f)
        d2Vee_df2 = self.coordinates.Vee_hessian_radials(angles, f)
        d2Vee_df_dangles = self.coordinates.Vee_hessian_angles_radials(angles, f)

        # Solve for dangles_df using reverse mode automatic differentiation
        dangles_df = jacrev(self._run_angles_only, argnums=1)(angles, f)
        return (
            jnp.sum(d2f * dVee_df)
            + df.dot(d2Vee_df2).dot(df)
            + df.dot(d2Vee_df_dangles.T.dot(dangles_df.dot(df)))
        )

    def d2Vee(
        self, angles: jnp.ndarray, f: jnp.ndarray, df: jnp.ndarray, d2f: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the second derivative of Vee w.r.t. r.

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.
        df : jnp.ndarray (N_grid - 1, Nel)
            The derivative of the co-motion functions at the grid points.
        d2f : jnp.ndarray (N_grid - 1, Nel)
            The second derivative of the co-motion functions at the grid points.

        Returns
        -------
        d2Vee: jnp.ndarray (N_grid - 1)
            The second derivative of Vee w.r.t. r.
        """
        dVee_df = self.coordinates.Vee_grad_radials(angles, f)
        d2Vee_df2 = self.coordinates.Vee_hessian_radials(angles, f)
        d2Vee_df_dangles = self.coordinates.Vee_hessian_angles_radials(angles, f)
        d2Vee_dangles_dangles = self.coordinates.Vee_hessian_angles(angles, f)

        # Solve for dangles_df using implicit differentiation
        dangles_df = -jnp.linalg.solve(d2Vee_dangles_dangles, d2Vee_df_dangles)
        return (
            jnp.sum(d2f * dVee_df)
            + df.dot(d2Vee_df2).dot(df)
            + df.dot(d2Vee_df_dangles.T.dot(dangles_df.dot(df)))
        )

    def lowest_eigvalsh_hessian(self, angles: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """
        Obtain the lowest eigenvalue of the angular hessian.

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.

        Returns
        -------
        lowest_eigvalsh_hessian : jnp.ndarray (N_grid - 1)
            The lowest eigenvalue of the angular hessian.
        """
        return jnp.linalg.eigvalsh(
            vmap(self.coordinates.Vee_hessian_angles, in_axes=(0, 0))(angles, f)
        )[:, 0]

    def lowest_eigvalsh_hessian_last(
        self, angles_last: jnp.ndarray, f_last: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Obtain the lowest eigenvalue of the angular hessian for the last point.

        Parameters
        ----------
        angles_last : jnp.ndarray (x)
            The angles at the last grid point.
        f_last : jnp.ndarray (Nel-1)
            The co-motion functions at the last grid point.

        Returns
        -------
        lowest_eigvalsh_hessian_last : jnp.ndarray (1)
            The lowest eigenvalue of the angular hessian for the last point.
        """
        return jnp.linalg.eigvalsh(self.coordinates_last.Vee_hessian_angles(angles_last, f_last))[0]

    def return_unique_angles(
        self, angles: jnp.ndarray, Vee: jnp.ndarray, f: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Find unique angles and select the N_select lowest Vee values.

        Parameters
        ----------
        angles : jnp.ndarray (N_random, x)
            The angles at the grid point obtained from random sampling.
        Vee : jnp.ndarray (N_random)
            The value of Vee at the grid point obtained from random sampling.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.

        Returns
        -------
        angles : jnp.ndarray (N_select, x)
            The unique angles at the grid point.
        Vee : jnp.ndarray (N_select)
            The value of Vee at the grid point.
        """
        Nel = f.shape[-1]
        MAD = self.coordinates.crossdist_all(f, angles) / Nel
        match_mask = MAD < self.equal_tol
        match_mask = jnp.triu(match_mask, k=1)
        unique = jnp.where(jnp.logical_not(jnp.any(match_mask, axis=1)), size=self.N_select)
        return angles[unique], Vee[unique]

    def N_unique_angles(self, angles: jnp.ndarray, f: jnp.ndarray) -> int:
        """
        Find unique angles and return the amount of unique angles.

        Parameters
        ----------
        angles : jnp.ndarray (N_random, x)
            The angles at the grid point obtained from random sampling.
        Vee : jnp.ndarray (N_random)
            The value of Vee at the grid point obtained from random sampling.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.

        Returns
        -------
        N_unique: int
            The amount of unique angles.
        """
        Nel = f.shape[-1]
        MAD = self.coordinates.crossdist_all(f, angles) / Nel
        match_mask = MAD < self.equal_tol
        match_mask = jnp.triu(match_mask, k=1)
        return jnp.sum(jnp.logical_not(jnp.any(match_mask, axis=1)))

    def sort_angles_Vee(
        self, angles: jnp.ndarray, Vee: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sort Vee and angles.

        Parameters
        ----------
        angles : jnp.ndarray (..., x)
            The angles at the grid point.
        Vee : jnp.ndarray (...)
            The value of Vee at the grid point.

        Returns
        -------
        angles : jnp.ndarray (..., x)
            The sorted angles at the grid point.
        """
        sort_idx = jnp.argsort(Vee, axis=-1)
        return jnp.take_along_axis(angles, sort_idx[..., None], axis=-2), jnp.take_along_axis(
            Vee, sort_idx, axis=-1
        )


@dataclass_json
@dataclass(eq=False)
class TwoElectron:
    """
    Obtain the results for two electrons, where no optimization is needed.

    Attributes
    ----------
    N_grid : int
        The number of grid points to use.
    grid : str
        The grid to use, either equidistant on "r" or "Ne". Default is "r".
    """

    N_grid: int
    grid: str = "r"

    def __post_init__(self):
        self.coordinates = get_coordinate_system("reduced")

        # Jit and vmap the derivative functions
        self.dVee_vmap = jit(vmap(self.dVee, in_axes=(0, 0, 0)))
        self.d2Vee_vmap = jit(vmap(self.d2Vee, in_axes=(0, 0, 0, 0)))

    def run(self, density: Density, *_) -> "AngularOptimizationResult":
        """
        Run the optimization.

        Parameters
        ----------
        density : Density
            The density object.

        Returns
        -------
        result: AngularOptimizationResult
            The result of the optimization.
        """

        # Set up the grid, from the point where the density integrates to 1
        # to the point where the density integrates to 2.
        print("Setting up grid, calculating density and co-motion functions...")
        if self.grid == "Ne":
            Ne = jnp.linspace(1, 2, self.N_grid)
            r = density.invNe(Ne)
            f = jnp.stack([r, density.invNe(2 - Ne)], axis=-1)
        else:
            raise NotImplementedError(
                f"Grid {self.grid} not implemented for N=2 (r is not implemented since a2=inf)."
            )
            # r = jnp.linspace(density.a[1], density.a[2], self.N_grid)
            # Ne = density.Ne(r)
            # f = jnp.stack([r, density.invNe(2-Ne)], axis=-1)

        # Calculate density, derivatives and coNe integral
        print("Calculating density, derivatives and coNe integral...")
        rho = density.rho(r)
        drho = density.rho_deriv(r)
        d2rho = density.rho_deriv2(r)
        coNe = density.coNe(density.invNe(jnp.arange(density.Nel)))

        angles = jnp.repeat(jnp.pi, self.N_grid - 1).reshape(self.N_grid - 1, 1)
        angles_last = jnp.array([jnp.pi])
        Vee = self.coordinates.Vee(angles, f[:-1])
        Vee_last = jnp.zeros((1))
        Vee = jnp.concatenate((Vee, Vee_last))

        # Calculate derivatives w.r.t. r for interpolation
        # Derivatives don't make sense for the last point
        print("Calculating derivatives of co-motion functions...")
        if self.grid == "r":
            df = density.co_motion_functions_deriv(r, f)
            d2f = density.co_motion_functions_deriv2(r, f, df)
        elif self.grid == "Ne":
            df = density.co_motion_functions_deriv_Ne(Ne, f)
            d2f = density.co_motion_functions_deriv2_Ne(Ne, f, df)
        else:
            raise NotImplementedError(f"Grid {self.grid} not implemented.")

        print("Calculating first derivative of Vee...")
        dVee = self.dVee_vmap(angles, f[:-1], df[:-1])
        print("Calculating second derivative of Vee...")
        d2Vee = self.d2Vee_vmap(angles, f[:-1], df[:-1], d2f[:-1])
        print("End of optimization.")

        return AngularOptimizationResult(
            opt=self,
            density=density,
            r=r,
            rho=rho,
            drho=drho,
            d2rho=d2rho,
            Ne=Ne,
            f=f,
            df=df,
            d2f=d2f,
            coNe=coNe,
            angles=angles,
            angles_last=angles_last,
            Vee=Vee,
            dVee=dVee,
            d2Vee=d2Vee,
        )

    def dVee(self, angles: jnp.ndarray, f: jnp.ndarray, df: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate the derivative of Vee w.r.t. r or Ne.

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.
        df : jnp.ndarray (N_grid - 1, Nel)
            The derivative of the co-motion functions at the grid points.

        Returns
        -------
        dVee: jnp.ndarray (N_grid - 1)
            The derivative of Vee w.r.t. r or Ne.
        """
        dVee_df = self.coordinates.Vee_grad_radials(angles, f)
        return jnp.sum(df * dVee_df)

    def d2Vee(
        self, angles: jnp.ndarray, f: jnp.ndarray, df: jnp.ndarray, d2f: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the second derivative of Vee w.r.t. r.

        Parameters
        ----------
        angles : jnp.ndarray (N_grid - 1, x)
            The angles at the grid points.
        f : jnp.ndarray (N_grid - 1, Nel)
            The co-motion functions at the grid points.
        df : jnp.ndarray (N_grid - 1, Nel)
            The derivative of the co-motion functions at the grid points.
        d2f : jnp.ndarray (N_grid - 1, Nel)
            The second derivative of the co-motion functions at the grid points.

        Returns
        -------
        d2Vee: jnp.ndarray (N_grid - 1)
            The second derivative of Vee w.r.t. r.
        """
        dVee_df = self.coordinates.Vee_grad_radials(angles, f)
        d2Vee_df2 = self.coordinates.Vee_hessian_radials(angles, f)

        # Solve for dangles_df using implicit differentiation
        return jnp.sum(d2f * dVee_df) + df.dot(d2Vee_df2).dot(df)


@dataclass(eq=False)
class AngularOptimizationResult:
    """
    Result of the angular optimization.

    Attributes
    ----------
    opt : AngularOptimization
        The AngularOptimization object that generated the result
    local_minimum : jnp.ndarray (N_grid - 1)
        Whether or not the angles at the grid point represent a local minimum.
    r : jnp.ndarray (N_grid)
        The radial coordinates.
    rho : jnp.ndarray (N_grid)
        The density.
    drho : jnp.ndarray (N_grid)
        The derivative of the density.
    d2rho : jnp.ndarray (N_grid)
        The second derivative of the density.
    Ne : jnp.ndarray (N_grid)
        The values of the cumulant at the grid points.
    f : jnp.ndarray (N_grid, Nel)
        The co-motion functions at the grid points.
    df : jnp.ndarray (N_grid, Nel)
        The derivative of the co-motion functions at the grid points.
    d2f : jnp.ndarray (N_grid, Nel)
        The second derivative of the co-motion functions at the grid points.
    coNe : jnp.ndarray (N_grid)
        The values of the co-cumulant at the grid points.
    angles : jnp.ndarray (N_grid - 1, x)
        The optimized angles at the grid points.
    Vee : jnp.ndarray (N_grid)
        The optimized value of Vee at the grid points.
    dVee : jnp.ndarray (N_grid-1)
        The derivative of Vee at the grid points.
    d2Vee : jnp.ndarray (N_grid-1)
        The second derivative of Vee at the grid points.
    min_starting_point : jnp.ndarray (N_grid - 1)
        Which starting point yielded the lowest Vee value for the grid point.
    min_random : jnp.ndarray (N_grid - 1)
        Which random optimized point yielded the lowest Vee value for the grid point.
    local_minimum_last : Optional[jnp.ndarray (1)]
        Whether or not the angles at the last grid point represent a local minimum.
    angles_last : Optional[jnp.ndarray (x,)]
        The optimized angles at the last grid point.
    """

    opt: AngularOptimization
    density: Density
    r: jnp.ndarray
    rho: jnp.ndarray
    drho: jnp.ndarray
    d2rho: jnp.ndarray
    Ne: jnp.ndarray
    f: jnp.ndarray
    df: jnp.ndarray
    d2f: jnp.ndarray
    coNe: jnp.ndarray
    angles: jnp.ndarray
    Vee: jnp.ndarray
    dVee: jnp.ndarray
    d2Vee: jnp.ndarray
    local_minimum: Optional[jnp.ndarray] = None
    min_starting_point: Optional[jnp.ndarray] = None
    min_random: Optional[jnp.ndarray] = None
    n_start: Optional[jnp.ndarray] = None
    local_minimum_last: Optional[jnp.ndarray] = None
    angles_last: Optional[jnp.ndarray] = None

    def save(self, filename: str, path: str = None):
        """
        Save the results to filename.json and filename.npz in path.

        Parameters
        ----------
        filename : str
            The name of the files to save the results to. (filename.json and filename.npz)
        path : str, optional
            The path to save the files to.
        """
        if path is not None:
            filename = os.path.join(path, filename)
        opt_dict = self.opt.to_dict()
        density_dict = self.density.encode()
        with open(filename + ".json", "w", encoding="utf-8") as f:
            json.dump({"opt": opt_dict, "density": density_dict}, f)
        jnp.savez(filename + ".npz", **self.to_dict())

    def to_dict(self) -> Dict[str, jnp.ndarray]:
        """
        Convert the result to a dictionary.

        Removes the optimization object.

        Returns
        -------
        dictionary : Dict[str, jnp.ndarray]
            The dictionary containing all the attributes except the optimization object.
        """
        dictionary: Dict[str, jnp.ndarray] = {}
        for field_name in self.__dataclass_fields__:
            if field_name not in ("opt", "density"):
                dictionary[field_name] = getattr(self, field_name)
        return dictionary

    @classmethod
    def load(cls, filename: str, path: str = None, **kwargs):
        """
        Load the results from filename.json and filename.npz in path.

        Parameters
        ----------
        filename : str
            The name of the files to load the results from. (filename.json and filename.npz)
        path : str, optional
            The path to load the files from.

        Returns
        -------
        result : AngularOptimizationResult
            The result loaded from the files.
        """
        if path is not None:
            filename = os.path.join(path, filename)
        with open(filename + ".json", "r", encoding="utf-8") as f:
            data = json.load(f)
        density = (
            get_density(**data["density"], **kwargs["density"])
            if "density" in kwargs
            else get_density(**data["density"])
        )
        if density.Nel == 2:
            opt = (
                TwoElectron(**data["opt"], **kwargs["opt"])
                if "opt" in kwargs
                else TwoElectron(**data["opt"])
            )
        else:
            coordinates = (
                get_coordinate_system(**data["opt"]["coordinates"], **kwargs["coordinates"])
                if "coordinates" in kwargs
                else get_coordinate_system(**data["opt"]["coordinates"])
            )
            coordinates_last = (
                get_coordinate_system(
                    **data["opt"]["coordinates_last"], **kwargs["coordinates_last"]
                )
                if "coordinates_last" in kwargs
                else get_coordinate_system(**data["opt"]["coordinates_last"])
            )
            data["opt"]["coordinates"] = coordinates
            data["opt"]["coordinates_last"] = coordinates_last
            opt = (
                AngularOptimization(**data["opt"], **kwargs["opt"])
                if "opt" in kwargs
                else AngularOptimization(**data["opt"])
            )
        data = jnp.load(filename + ".npz", allow_pickle=True)
        return cls(
            opt=opt,
            density=density,
            **data,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in ["density", "opt", "coordinates", "coordinates_last"]
            },
        )

    @classmethod
    def to_numpy(cls, result: "AngularOptimizationResult") -> "AngularOptimizationResult":
        """
        Convert the result to numpy arrays.

        Parameters
        ----------
        result : AngularOptimizationResult
            The result to convert.

        Returns
        -------
        result : AngularOptimizationResult
            The result with all the attributes converted to numpy arrays.
        """
        result_dict = result.to_dict()
        for key, value in result_dict.items():
            if isinstance(value, jnp.ndarray):
                result_dict[key] = np.array(value)
        return cls(opt=result.opt, density=result.density, **result_dict)
