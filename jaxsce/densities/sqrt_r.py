"""SqrtR density class."""

import jax
import jax.numpy as jnp
import scipy.special

# Import inverse gamma from TF, doesn't seem to be available anywhere else
from tensorflow_probability.substrates.jax.math import igammacinv

from .classes import Density


class SqrtR(Density):
    """
    SqrtR density class, corresponding to rho(r) proportional to sqrt(r) exp(-r).

    Additional methods:
    ----------------------
    invNe_deriv_manual:
        Derivative of the inverse of Ne, simplified, but the invNe in the base class is fine.
    """

    __doc__ = Density.__doc__ + __doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "sqrt_r"

        # Integrals were derived analytically in Mathematica
        self.U = self.Nel**2 * (15 * jnp.pi - 16) / (75 * jnp.pi)
        self.LDA_int = (
            9
            * (3 / 5) ** (1 / 3)
            * self.Nel ** (4 / 3)
            * scipy.special.gamma(11 / 3)
            / (80 * jnp.pi)
        )
        self.GEA_int = 21 * (3 / 10) ** (2 / 3) * self.Nel ** (2 / 3) * scipy.special.gamma(4 / 3)

        self.__post_init__()
        self.add_jvps()
        self.add_jit()

    # Expressions for functions were derived analytically in Mathematica

    def rho(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * 2 * jnp.sqrt(r) * jnp.exp(-r) / (15 * jnp.pi ** (3 / 2))

    def rho_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * (1 - 2 * r) * jnp.exp(-r) / (jnp.sqrt(r) * 15 * jnp.pi ** (3 / 2))

    def rho_deriv2(self, r: jnp.ndarray) -> jnp.ndarray:
        return (
            self.Nel * (4 * r * (r - 1) - 1) * jnp.exp(-r) / (r ** (3 / 2) * 30 * jnp.pi ** (3 / 2))
        )

    def Ne(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * (1 - jax.scipy.special.gammaincc(7 / 2, r))

    def coNe(self, r: jnp.ndarray) -> jnp.ndarray:
        return (
            2
            / 15
            * self.Nel
            * (
                2 * jnp.sqrt(r) * jnp.exp(-r) * (2 * r + 3) / (jnp.sqrt(jnp.pi))
                + 3 * jax.scipy.special.erfc(jnp.sqrt(r))
            )
        )

    def invNe(self, n: jnp.ndarray) -> jnp.ndarray:
        return igammacinv(7 / 2, 1 - n / self.Nel)

    def invNe_deriv_manual(self, n: jnp.ndarray) -> jnp.ndarray:
        """
        This is the derivative of the invNe function,
        simplified, but the invNe in the base class is fine

        Parameters:
        ----------
        n: jnp.ndarray
            Ne evaluated at some r.

        Returns:
        -------
        invNe_deriv: jnp.ndarray
            Derivative of the inverse of Ne.
        """
        # T
        return (
            (15 * jnp.sqrt(jnp.pi) * jnp.exp(igammacinv(7 / 2, 1 - self.Nel / n)))
            / (8 * igammacinv(7 / 2, 1 - self.Nel / n))
            / self.Nel
        )
