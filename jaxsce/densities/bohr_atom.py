"""This module contains the class for the Bohr atom density"""

from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import sympy
import sympy2jax
from pyscf import dft
from sympy.physics.hydrogen import R_nl

from .classes import NumericalInvNeDensity

sympy_functions = ["rho", "rho_deriv", "rho_deriv2", "Ne", "Ne_deriv", "coNe", "coNe_deriv", "vH"]


class SympyDensity(NumericalInvNeDensity):
    """
    Density class that uses sympy to compute the functions.

    Additional attributes:
    ----------------------
    N_int: int
        Number of points to use for numerical integration of the LDA and GEA integrals.
    rho_trunc: float
        Truncation value for the density.

    Additional methods:
    ----------------------
    x_sympy: sympy.Expr
        Sympy expression for the object x.
    x_mod: sympy2jax.SymbolicModule
        Module that converts the sympy expression to a jax function.
    """

    __doc__ = NumericalInvNeDensity.__doc__ + __doc__

    def __init__(
        self,
        r_symbol: sympy.Symbol,
        rho_sympy: sympy.Expr,
        N_int: int = 100000,
        rho_trunc: float = 0.0,
        scale_int: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Get all the functions we need in sympy and convert to jax functions via symp2yjax
        self.N_int = N_int
        self.rho_trunc = rho_trunc
        self.scale_int = scale_int

        self.rho_sympy = rho_sympy
        self.rho_deriv_sympy = self.rho_sympy.diff(r_symbol)
        self.rho_deriv2_sympy = self.rho_deriv_sympy.diff(r_symbol)
        self.Ne_sympy = sympy.integrate(self.rho_sympy * r_symbol**2, (r_symbol, 0, r_symbol))
        self.Ne_deriv_sympy = self.rho_sympy * r_symbol**2
        self.coNe_sympy = sympy.integrate(self.rho_sympy * r_symbol, (r_symbol, r_symbol, sympy.oo))
        self.coNe_deriv_sympy = -self.rho_sympy * r_symbol
        self.vH_sympy = self.Ne_sympy / r_symbol + self.coNe_sympy
        self.U = float(
            sympy.integrate(
                self.rho_sympy * r_symbol**2 * self.vH_sympy / 2,
                (r_symbol, 0, sympy.oo),
            )
        )
        for function in sympy_functions:
            mod_function = sympy2jax.SymbolicModule(getattr(self, function + "_sympy"))
            setattr(
                self,
                function + "_mod",
                mod_function,
            )

        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        # Set up grid to compute integrals LDA and GEA integrals
        r_int, dr_int = dft.radi.treutler(self.N_int)
        r_int = r_int * self.scale_int
        dr_int = dr_int * self.scale_int
        self.LDA_int = float(jnp.sum(dr_int * 4 * jnp.pi * r_int**2 * self.rho(r_int) ** (4 / 3)))
        self.Nel_int = float(jnp.sum(dr_int * 4 * jnp.pi * r_int**2 * self.rho(r_int)))
        non_zero = jnp.where(self.rho(r_int) > self.rho_trunc)
        r_int = r_int[non_zero]
        dr_int = dr_int[non_zero]
        self.GEA_int = float(
            jnp.sum(
                dr_int
                * 4
                * jnp.pi
                * r_int**2
                * self.rho_deriv(r_int) ** 2
                / (self.rho(r_int) ** (4 / 3))
            )
        )

    def encode(self) -> Dict[str, Union[str, int, float]]:
        encode_dict = super().encode()
        encode_dict.update(
            {"N_int": self.N_int, "rho_trunc": self.rho_trunc, "scale_int": self.scale_int}
        )
        return encode_dict

    def rho(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.rho_mod(r=r, pi=jnp.pi) / (4 * jnp.pi)

    def Ne(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Ne_mod(r=r, pi=jnp.pi)

    def coNe(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.coNe_mod(r=r, pi=jnp.pi)

    def vH(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.vH_mod(r=r, pi=jnp.pi)

    def rho_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.rho_deriv_mod(r=r, pi=jnp.pi) / (4 * jnp.pi)

    def rho_deriv2(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.rho_deriv2_mod(r=r, pi=jnp.pi) / (4 * jnp.pi)

    def Ne_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Ne_deriv_mod(r=r, pi=jnp.pi)

    def coNe_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.coNe_deriv_mod(r=r, pi=jnp.pi)


class BohrAtom(SympyDensity):
    """
    Bohr atom density.
    Computes the density of a Bohr atom with Nel electrons by filling (sub)shells sequentially.
    """

    __doc__ = SympyDensity.__doc__ + __doc__

    def __init__(self, Nel: int = 0, scale_int: float = 3.0, **kwargs):
        if Nel == 0:
            raise ValueError("Nel is not specified or equal to 0.")

        # Based on the number of electons, determine the subshells to fill.
        nl_pairs: List[Tuple[int, int]] = []
        Nel_count = 0
        found_Nel = False
        for n in range(1, 6):
            for l in range(n):
                nl_pairs.append((n, l))
                Nel_count += 2 * (2 * l + 1)
                if Nel_count == Nel:
                    found_Nel = True
                    break
            if found_Nel:
                break
        if not found_Nel:
            raise ValueError(f"{Nel} does not fit a Bohr Atom with subshells filled sequentially.")

        # Build the density using sympy
        rho_sympy = 0
        r_symbol = sympy.Symbol("r", positive=True)
        for n, l in nl_pairs:
            rho_sympy += 2 * (2 * l + 1) * R_nl(n, l, r_symbol) ** 2
        super().__init__(r_symbol, rho_sympy, Nel=Nel, scale_int=scale_int, **kwargs)
        self.name = "bohr_atom"
