"""Test densities"""
from jax import numpy as jnp
from jax.config import config
from jax.scipy.special import erfc

from .classes import Density, NumericalInvNeDensity

config.update("jax_enable_x64", True)


class ExponentialDensity(NumericalInvNeDensity):
    """Exponential density"""

    def __init__(
        self,
        Nel: int = 10,
        cumulant_threshold: float = 1e-8,
        N_points_Ne: int = 1000,
        invNe_tol: float = 1e-10,
    ):
        super().__init__(Nel, cumulant_threshold, N_points_Ne, invNe_tol)
        self.__post_init__()

    def rho(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * jnp.exp(-r) / (8 * jnp.pi)

    def Ne(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * (1 - jnp.exp(-r) * (r * (r + 2) + 2) / 2)

    def coNe(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel / 2 * jnp.exp(-r) * (r + 1)


class InvRGaussian(Density):
    """Inv R Gaussian"""

    def __init__(self, Nel: int = 10):
        super().__init__(Nel)
        self.add_jvps(also_rho=False)
        self.add_jit()
        self.__post_init__()

    def rho(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * jnp.exp(-(r**2) / 2) / (r * 4 * jnp.pi)

    def Ne(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * (1 - jnp.exp(-(r**2) / 2))

    def invNe(self, n: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(2 * jnp.log(self.Nel / (self.Nel - n)))

    def coNe(self, r: jnp.ndarray) -> jnp.ndarray:
        return self.Nel * jnp.sqrt(jnp.pi / 2) * erfc(r / jnp.sqrt(2))
