"""Base classes for the spherically symmetric electron densities."""

import abc
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import custom_jvp, grad, jit, lax, vmap


def add_jvp(
    function: Callable[[jnp.ndarray], jnp.ndarray],
    function_deriv: Callable[[jnp.ndarray], jnp.ndarray],
):
    """
    Add a jvp to a function.

    Parameters
    ----------
    function : function
        Function to add a jvp to.
    function_deriv : function
        Derivative of the function.

    Returns
    -------
    function : function
        Function with a jvp.
    """

    function_jvp = custom_jvp(function)

    @function_jvp.defjvp
    def jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        return function(x), function_deriv(x) * x_dot

    return function_jvp


def add_jvp_share(
    function: Callable[[Union[jnp.ndarray, float]], Union[jnp.ndarray, float]],
    function_deriv: Callable[
        [Union[jnp.ndarray, float], Union[jnp.ndarray, float]], Union[jnp.ndarray, float]
    ],
):
    """
    Add a jvp to a function. The function value is shared with the derivative.

    Parameters
    ----------
    function : function
        Function to add a jvp to.
    function_deriv : function
        Derivative of the function.

    Returns
    -------
    function : function
        Function with a jvp.
    """

    function_jvp = custom_jvp(function)

    @function_jvp.defjvp
    def jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        y = function(x)
        return y, function_deriv(x, y) * x_dot

    return function_jvp


# Lits of functions to jit in density.add_jit()
jit_functions = [
    "rho",
    "rho_deriv",
    "rho_deriv2",
    "Ne",
    "Ne_deriv",
    "Ne_deriv2",
    "coNe",
    "coNe_deriv",
    "invNe",
    "invNe_deriv",
    "co_motion_function_Ne",
    "co_motion_functions",
    "co_motion_function",
    "co_motion_functions_Ne",
    "co_motion_functions_deriv",
    "co_motion_function_deriv",
    "co_motion_functions_deriv_Ne",
    "co_motion_function_deriv_Ne",
    "co_motion_function_deriv2",
    "co_motion_functions_deriv2",
    "co_motion_function_deriv2_Ne",
    "co_motion_functions_deriv2_Ne",
    "vH",
    "vH_deriv",
]


class Density(metaclass=abc.ABCMeta):
    """
    Base class for the electron densities.

    Attributes
    ----------
    Nel : int
        Number of electrons.
    a : (N,) array
        Points at which the density integrates to an integer (ranging from 0 to N-1).
    LDA_int : float
        Integral of rho(r)**4/3.
    GEA_int : float
        Integral of (d rho(r)/dr)**2/rho(r)**4/3.

    Methods
    -------
    add_jvps(also_rho=True)
        Add jvps to the functions.
    add_jit()
        Jit the functions in the jit_functions list.
    encode()
        Encode the density as a dictionary to store.
    rho(r)
        Compute the electron density at r.
    rho_deriv(r)
        Compute the derivative of the electron density at r.
    rho_deriv2(r)
        Compute the second derivative of the electron density at r.
    Ne(r)
        Compute the electron density cumulant at r.
    Ne_deriv(r)
        Compute the derivative of the electron density cumulant at r.
    Ne_deriv2(r)
        Compute the second derivative of the electron density cumulant at r.
    coNe(r)
        Compute the integral of rho(r)/r from r to infinity.
    coNe_deriv(r)
        Compute the derivative of the integral of rho(r)/r from r to infinity.
    invNe(n)
        Compute the inverse of the electron density cumulant at n.
    invNe_deriv(n, invNe(n)=None)
        Compute the derivative of the inverse of the electron density cumulant at n.
    co_motion_function(r)
        Compute the co-motion functions at a single r.
    co_motion_functions(r)
        Compute the co-motion functions at r.
    co_motion_function_Ne(Ne(r))
        Compute the co-motion functions at a single r, using Ne(r) instead of r.
    co_motion_functions_Ne(Ne(r))
        Compute the co-motion functions at r, using Ne(r) instead of r.
    co_motion_function_deriv(r)
        Compute the derivative of the co-motion functions at a single r.
    co_motion_functions_deriv(r)
        Compute the derivative of the co-motion functions at r.
    co_motion_function_deriv_Ne(Ne)
        Compute the derivative of the co-motion functions at a single r, using Ne(r) instead of r.
    co_motion_functions_deriv_Ne(Ne)
        Compute the derivative of the co-motion functions at r, using Ne(r) instead of r.
    co_motion_function_deriv2(r)
        Compute the second derivative of the co-motion functions at a single r.
    co_motion_functions_deriv2(r)
        Compute the second derivative of the co-motion functions at r.
    co_motion_function_deriv2_Ne(Ne)
        Compute the second derivative of the co-motion functions at a single r,
        using Ne(r) instead of r.
    co_motion_functions_deriv2_Ne(Ne)
        Compute the second derivative of the co-motion functions at r, using Ne(r) instead of r.
    vH(r)
        Compute the Hartree potential at r.
    vH_deriv(r)
        Compute the derivative of the Hartree potential at r.
    """

    def __init__(self, Nel: int = 0):
        """
        Initialize the density.

        Parameters
        ----------
        Nel : int
            Number of electrons.
        """
        self.name: str = ""
        if Nel == 0:
            raise ValueError("Nel is not specified or is 0.")
        self.Nel: int = Nel
        self.a: jnp.ndarray = jnp.empty(Nel)
        self.LDA_int: float = 0.0
        self.GEA_int: float = 0.0

        if self.Nel % 2 == 0:
            self.co_motion_function = self.co_motion_function_even
            self.co_motion_function_Ne = self.co_motion_function_even_Ne
            self.co_motion_functions = self.co_motion_functions_even
            self.co_motion_functions_Ne = self.co_motion_functions_even_Ne
        else:
            self.co_motion_function = self.co_motion_function_odd
            self.co_motion_function_Ne = self.co_motion_function_odd_Ne
            self.co_motion_functions = self.co_motion_functions_odd
            self.co_motion_functions_Ne = self.co_motion_functions_odd_Ne

    def __post_init__(self):
        """
        After initialization,
        calculate the points at which the density integrates to an integer.
        """
        self.a = self.invNe(jnp.arange(self.Nel))

    def add_jvps(self, also_rho: bool = True):
        """
        Add jvps to the functions for which this applies.

        Parameters
        ----------
        also_rho : bool
            Whether to add jvps to the rho function as well, by default True

        Functions are: Ne, coNe, rho, invNe,
        co_motion_function, co_motion_functions,
        co_motion_function_Ne, co_motion_functions_Ne
        """
        self.Ne = add_jvp(self.Ne, self.Ne_deriv)
        self.coNe = add_jvp(self.coNe, self.coNe_deriv)
        if also_rho:
            self.rho = add_jvp(self.rho, self.rho_deriv)
        self.invNe = add_jvp_share(self.invNe, self.invNe_deriv)
        self.co_motion_function = add_jvp_share(
            self.co_motion_function, self.co_motion_function_deriv
        )
        self.co_motion_functions = add_jvp_share(
            self.co_motion_functions, self.co_motion_functions_deriv
        )
        self.co_motion_function_Ne = add_jvp_share(
            self.co_motion_function_Ne, self.co_motion_function_deriv_Ne
        )
        self.co_motion_functions_Ne = add_jvp_share(
            self.co_motion_functions_Ne, self.co_motion_functions_deriv_Ne
        )

    def add_jit(self):
        """
        Jit the functions listed in the list jit_functions.
        """
        for function in jit_functions:
            setattr(self, function, jit(getattr(self, function)))

    def encode(self) -> dict:
        """
        Encode the density as a dictionary to store in a json.

        Returns
        -------
        dict
            Dictionary containing the density parameters,
            must match the arguments of the corresponding density class.
        """
        return {"name": self.name, "Nel": self.Nel}

    @abc.abstractmethod
    def rho(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the electron density.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        rho : (N_points,) array
            Density at the given radial coordinates.
        """

    def rho_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the derivative of the electron density.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        rho_deriv : (N_points,) array
            Derivative of the electron density at the given radial coordinates.
        """
        return vmap(grad(self.rho), in_axes=0)(r.reshape(-1)).reshape(r.shape)

    def rho_deriv2(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the second derivative of the electron density.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        rho_deriv2 : (N_points,) array
            Second derivative of the electron density at the given radial coordinates.
        """
        return vmap(grad(grad(self.rho)), in_axes=0)(r.reshape(-1)).reshape(r.shape)

    @abc.abstractmethod
    def Ne(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the cumulant of the electron density.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        Ne : (N_points,) array
            Cumulant of the electron density.
        """

    def Ne_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the derivative of the cumulant of the radial density of the electrons.

        Known analytically from the definition of Ne(r)

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        Ne_deriv : (N_points,) array
            Derivative of the cumulant of the electron density.
        """
        return 4 * jnp.pi * r**2 * self.rho(r)

    def Ne_deriv2(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the second derivative of the cumulant of the electron density.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        Ne_deriv2 : (N_points,) array
            Second derivative of the cumulant of the electron density.
        """
        return 8 * jnp.pi * r * self.rho(r) + 4 * jnp.pi * r**2 * self.rho_deriv(r)

    @abc.abstractmethod
    def coNe(self, r: jnp.ndarray) -> jnp.ndarray:
        """Compute the co-Cumulant (integral of rho(r)/r from r to infinity)

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        coNe : (N_points,) array
            Co-Cumulant.
        """

    def coNe_deriv(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the derivative of the co-Cumulant (integral of rho(r)/r from r to infinity)

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        coNe_deriv : (N_points,) array
            Derivative of the co-Cumulant.
        """
        return -4 * jnp.pi * r * self.rho(r)

    @abc.abstractmethod
    def invNe(self, n: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the inverse of the cumulant of the electron density.

        Parameters
        ----------
        n : (N_points,) array
            Cumulant of the electron density.

        Returns
        -------
        r : (N_points,) array
            Radial coordinate.
        """

    def invNe_deriv(self, n: jnp.ndarray, r: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute the derivative of the inverse
        of the cumulant of the electron density.

        Parameters
        ----------
        n : (N_points,) array
            Cumulant of the electron density.
        r : (N_points,) array
            Radial coordinates (invNe(n)), if already computed. Default is no.

        Returns
        -------
        invNe_deriv : (N_points,) array
            Derivative of the inverse of the cumulant of the electron density.
        """
        if r is None:
            r = self.invNe(n)
        return 1 / (self.Ne_deriv(r))

    def invNe_deriv2(
        self,
        n: jnp.ndarray,
        r: Optional[jnp.ndarray] = None,
        invNe_deriv: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute the second derivative of the inverse
        of the cumulant of the electron density.

        Parameters
        ----------
        n : (N_points,) array
            Cumulant of the electron density.
        r : (N_points,) array
            Radial coordinates (invNe(n)), if already computed. Default is no.
        invNe_deriv : (N_points,) array
            Derivative of the inverse of the cumulant of the electron density,

        Returns
        -------
        invNe_deriv2 : (N_points,) array
            Second derivative of the inverse of the cumulant of the electron density.
        """
        if r is None:
            r = self.invNe(n)
        if invNe_deriv is None:
            invNe_deriv = self.invNe_deriv(n, r)
        return -self.Ne_deriv2(r) / (self.Ne_deriv(r) ** 2) * invNe_deriv

    def co_motion_function_odd(self, r: float) -> jnp.ndarray:
        """
        Compute the co-motion functions for an odd number of electrons.

        Parameters
        ----------
        r : float
            Radial coordinat of the first electron

        Returns
        -------
        f : (Nel) array
            Co-motion functions for an odd number of electrons.
        """

        f = jnp.zeros(self.Nel).at[0].set(r)
        Nmax = (self.Nel + 1) // 2

        Ne = self.Ne(r)

        k = jnp.arange(1, Nmax)
        f = f.at[1::2].set(self.invNe(jnp.sign(2 * k - Ne) * (2 * k - Ne)))
        f = f.at[2::2].set(
            self.invNe(
                jnp.sign(self.Nel - 2 * k - Ne) * (Ne + 2 * k - self.Nel) + self.Nel,
            )
        )
        return f

    def co_motion_function_odd_Ne(self, Ne: float) -> jnp.ndarray:
        """Compute the co-motion functions for an odd number of electrons.

        Parameters
        ----------
        Ne : float
            Cumulant of the radial density of the electrons at the position of electron 1.

        Returns
        -------
        f : (Nel) array
            Co-motion functions for an odd number of electrons.
        """

        f = jnp.zeros(self.Nel).at[0].set(self.invNe(Ne))
        Nmax = (self.Nel + 1) // 2

        k = jnp.arange(1, Nmax)
        f = f.at[1::2].set(self.invNe(jnp.sign(2 * k - Ne) * (2 * k - Ne)))
        f = f.at[2::2].set(
            self.invNe(
                jnp.sign(self.Nel - 2 * k - Ne) * (Ne + 2 * k - self.Nel) + self.Nel,
            )
        )
        return f

    def co_motion_functions_odd(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the co-motion functions for an odd number of electrons.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        f : (N_points, Nel) array
            Co-motion functions.
        """

        f = jnp.zeros((self.Nel, r.shape[0])).at[0].set(r)
        Nmax = (self.Nel + 1) // 2

        Ne = self.Ne(r)

        k = jnp.arange(1, Nmax)
        f = f.at[1::2].set(
            self.invNe(jnp.sign(2 * k[:, None] - Ne[None, :]) * (2 * k[:, None] - Ne[None, :]))
        )
        f = f.at[2::2].set(
            self.invNe(
                jnp.sign(self.Nel - 2 * k[:, None] - Ne[None, :])
                * (Ne[None, :] + 2 * k[:, None] - self.Nel)
                + self.Nel,
            )
        )
        return f.T

    def co_motion_functions_odd_Ne(self, Ne: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the co-motion functions for an odd number of electrons.

        Parameters
        ----------
        Ne : (N_points,) array
            Cumulant of the radial electron density (at the position of electron 1).

        Returns
        -------
        f : (N_points, Nel) array
            Co-motion functions.
        """

        f = jnp.zeros((self.Nel, Ne.shape[0])).at[0].set(self.invNe(Ne))
        Nmax = (self.Nel + 1) // 2

        k = jnp.arange(1, Nmax)
        f = f.at[1::2].set(
            self.invNe(jnp.sign(2 * k[:, None] - Ne[None, :]) * (2 * k[:, None] - Ne[None, :]))
        )
        f = f.at[2::2].set(
            self.invNe(
                jnp.sign(self.Nel - 2 * k[:, None] - Ne[None, :])
                * (Ne[None, :] + 2 * k[:, None] - self.Nel)
                + self.Nel,
            )
        )
        return f.T

    def co_motion_function_even(self, r: float) -> jnp.ndarray:
        """
        Compute the co-motion functions for an even number of electrons.

        Parameters
        ----------
        r : float
            Radial coordinate.

        Returns
        -------
        f : (N_el,) array
            Co-motion functions.
        """
        f = jnp.zeros(self.Nel).at[0].set(r)
        Nmax = (self.Nel) // 2

        Ne = self.Ne(r)
        k = jnp.arange(1, Nmax)
        f = f.at[1:-1:2].set(self.invNe(jnp.sign(2 * k - Ne) * (2 * k - Ne)))
        f = f.at[2::2].set(
            self.invNe(jnp.sign(self.Nel - 2 * k - Ne) * (Ne + 2 * k - self.Nel) + self.Nel)
        )
        f = f.at[-1].set(self.invNe(self.Nel - Ne))
        return f

    def co_motion_function_even_Ne(self, Ne: float) -> jnp.ndarray:
        """Compute the co-motion functions for an even number of electrons.

        Parameters
        ----------
        Ne : float
            Cumulant of the electron density (at the position of electron 1).

        Returns
        -------
        f : (N_el,) array
            Co-motion functions.
        """
        f = jnp.zeros(self.Nel).at[0].set(self.invNe(Ne))
        Nmax = (self.Nel) // 2

        k = jnp.arange(1, Nmax)
        f = f.at[1:-1:2].set(self.invNe(jnp.sign(2 * k - Ne) * (2 * k - Ne)))
        f = f.at[2::2].set(
            self.invNe(jnp.sign(self.Nel - 2 * k - Ne) * (Ne + 2 * k - self.Nel) + self.Nel)
        )
        f = f.at[-1].set(self.invNe(self.Nel - Ne))
        return f

    def co_motion_functions_even(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the co-motion functions for an even number of electrons.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        f : (N_points, N_el) array
            Co-motion functions.
        """
        f = jnp.zeros((self.Nel, r.size)).at[0].set(r)
        Nmax = (self.Nel) // 2

        Ne = self.Ne(r)
        k = jnp.arange(1, Nmax)
        f = f.at[1:-1:2].set(
            self.invNe(jnp.sign(2 * k[:, None] - Ne[None, :]) * (2 * k[:, None] - Ne[None, :]))
        )
        f = f.at[2::2].set(
            self.invNe(
                jnp.sign(self.Nel - 2 * k[:, None] - Ne[None, :])
                * (Ne[None, :] + 2 * k[:, None] - self.Nel)
                + self.Nel
            )
        )
        f = f.at[-1].set(self.invNe(self.Nel - Ne))
        return f.T

    def co_motion_functions_even_Ne(self, Ne: jnp.ndarray) -> jnp.ndarray:
        """Compute the co-motion functions for an even number of electrons.

        Parameters
        ----------
        Ne : (N_points,) array
            Cumulant of the electron density (at the position of electron 1).

        Returns
        -------
        f : (N_points, N_el) array
            Co-motion functions.
        """
        f = jnp.zeros((self.Nel, Ne.size)).at[0].set(self.invNe(Ne))
        Nmax = (self.Nel) // 2

        k = jnp.arange(1, Nmax)
        f = f.at[1:-1:2].set(
            self.invNe(jnp.sign(2 * k[:, None] - Ne[None, :]) * (2 * k[:, None] - Ne[None, :]))
        )
        f = f.at[2::2].set(
            self.invNe(
                jnp.sign(self.Nel - 2 * k[:, None] - Ne[None, :])
                * (Ne[None, :] + 2 * k[:, None] - self.Nel)
                + self.Nel
            )
        )
        f = f.at[-1].set(self.invNe(self.Nel - Ne))
        return f.T

    def co_motion_function_deriv(self, r: float, f: jnp.ndarray = None) -> jnp.ndarray:
        """
        Compute the derivative of the co-motion functions.

        Parameters
        ----------
        r : float
            Radial coordinate.
        f : (Nel,) array
            Co-motion functions if already computed. Default is no.

        Returns
        -------
        f_deriv : (Nel,) array
            Derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_function(r)
        f_deriv = jnp.empty(self.Nel).at[0].set(1.0)
        Nmax = (self.Nel + self.Nel % 2) // 2
        k = jnp.arange(1, Nmax)
        f_deriv = f_deriv.at[1:-1:2].set(
            jnp.sign(self.a[2 * k] - r)
            * r**2
            / f[1:-1:2] ** 2
            * self.rho(r)
            / self.rho(f[1:-1:2])
        )
        f_deriv = f_deriv.at[2::2].set(
            jnp.sign(self.a[self.Nel - 2 * k] - r)
            * r**2
            / f[2::2] ** 2
            * self.rho(r)
            / self.rho(f[2::2])
        )
        if self.Nel % 2 == 0:
            f_deriv = f_deriv.at[-1].set(-(r**2) / f[-1] ** 2 * self.rho(r) / self.rho(f[-1]))

        return f_deriv

    def co_motion_function_deriv2(
        self, r: float, f: Optional[jnp.ndarray] = None, f_deriv: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the second derivative of the co-motion functions.

        Parameters
        ----------
        r : float
            Radial coordinates of the N electrons.
        f : (Nel,) array
            Co-motion functions if already computed. Default is no.
        f_deriv : (Nel,) array
            Derivative of the co-motion functions if already computed. Default is no.
        Returns
        -------
        f_deriv2 : (Nel,) array
            Second erivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_function(r)
        if f_deriv is None:
            f_deriv = self.co_motion_function_deriv(r, f)
        return (-1) ** jnp.arange(self.Nel) * (
            2 * r * self.rho(r) / (f**2 * self.rho(f))
            - 2 * r**2 * self.rho(r) * f_deriv / (f**3 * self.rho(f))
            + r**2 * self.rho_deriv(r) / (f**2 * self.rho(f))
            - r**2 * self.rho(r) * f_deriv * self.rho_deriv(f) / (f**2 * self.rho(f) ** 2)
        )

    def co_motion_function_deriv_Ne(
        self, Ne: float, f: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the derivative of the co-motion functions.

        Parameters
        ----------
        Ne : float
            Cumulant of the electron density (at the position of electron 1).
        f : (Nel) array
            Co-motion functions if already computed. Default is no.

        Returns
        -------
        f_deriv : (N_points, Nel) array
            Derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_function_Ne(Ne)
        f_deriv = jnp.empty(self.Nel).at[0].set(self.invNe_deriv(Ne))
        Nmax = (self.Nel + self.Nel % 2) // 2
        k = jnp.arange(1, Nmax)
        f_deriv = f_deriv.at[1:-1:2].set(
            jnp.sign(2 * k - Ne) / (4 * jnp.pi * f[1:-1:2] ** 2 * self.rho(f[1:-1:2]))
        )
        f_deriv = f_deriv.at[2::2].set(
            jnp.sign(self.Nel - 2 * k - Ne) / (4 * jnp.pi * f[2::2] ** 2 * self.rho(f[2::2]))
        )
        if self.Nel % 2 == 0:
            f_deriv = f_deriv.at[-1].set(-1 / (4 * jnp.pi * f[-1] ** 2 * self.rho(f[-1])))
        return f_deriv

    def co_motion_function_deriv2_Ne(
        self, Ne: float, f: Optional[jnp.ndarray] = None, f_deriv: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the second derivative of the co-motion functions.

        Parameters
        ----------
        Ne : float
            Cumulant of the electron density (at the position of electron 1).
        f : (Nel,) array
            Co-motion functions if already computed. Default is no.
        f_deriv : (Nel,) array
            Derivative of the co-motion functions if already computed. Default is no.

        Returns
        -------
        f_deriv2 : (N_points, Nel) array
            Second derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_function_Ne(Ne)
        if f_deriv is None:
            f_deriv = self.co_motion_function_deriv_Ne(Ne, f)
        return (-1) ** jnp.arange(self.Nel) * (
            -2 * f_deriv / (4 * jnp.pi * f**3 * self.rho(f))
            - f_deriv * self.rho_deriv(f) / (4 * jnp.pi * f**2 * self.rho(f) ** 2)
        )

    def co_motion_functions_deriv(
        self, r: jnp.ndarray, f: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the derivative of the co-motion functions.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.
        f : (N_points, Nel) array
            Co-motion functions if already computed. Default is no.

        Returns
        -------
        f_deriv : (N_points, Nel) array
            Derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_functions(r)
        f_deriv = jnp.empty((*r.shape, self.Nel)).at[:, 0].set(1.0)
        Nmax = (self.Nel + self.Nel % 2) // 2
        k = jnp.arange(1, Nmax)
        f_deriv = f_deriv.at[:, 1:-1:2].set(
            -jnp.sign(self.a[2 * k[None, :]] - r[:, None])
            * r[:, None] ** 2
            / f[:, 1:-1:2] ** 2
            * self.rho(r[:, None])
            / self.rho(f[:, 1:-1:2])
        )
        f_deriv = f_deriv.at[:, 2::2].set(
            jnp.sign(self.a[self.Nel - 2 * k[None, :]] - r[:, None])
            * r[:, None] ** 2
            / f[:, 2::2] ** 2
            * self.rho(r[:, None])
            / self.rho(f[:, 2::2])
        )
        if self.Nel % 2 == 0:
            f_deriv = f_deriv.at[:, -1].set(
                -(r**2) / f[:, -1] ** 2 * self.rho(r) / self.rho(f[:, -1])
            )
        return f_deriv

    def co_motion_functions_deriv2(
        self, r: jnp.ndarray, f: Optional[jnp.ndarray] = None, f_deriv: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the second derivative of the co-motion functions.

        Parameters
        ----------
        r : (N_grid,) array
            Radial coordinate.
        f : (N_grid, Nel) array
            Co-motion functions if already computed. Default is no.
        f_deriv : (N_grid, Nel) array
            Derivative of the co-motion functions if already computed. Default is no.
        Returns
        -------
        f_deriv2 : (N_grid, Nel) array
            Second derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_functions(r)
        if f_deriv is None:
            f_deriv = self.co_motion_functions_deriv(r, f)
        return (-1) ** jnp.arange(self.Nel)[None, :] * (
            2 * r[:, None] * self.rho(r[:, None]) / (f**2 * self.rho(f))
            - 2 * r[:, None] ** 2 * self.rho(r[:, None]) * f_deriv / (f**3 * self.rho(f))
            + r[:, None] ** 2 * self.rho_deriv(r[:, None]) / (f**2 * self.rho(f))
            - r[:, None] ** 2
            * self.rho(r[:, None])
            * f_deriv
            * self.rho_deriv(f)
            / (f**2 * self.rho(f) ** 2)
        )

    def co_motion_functions_deriv_Ne(
        self, Ne: jnp.ndarray, f: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute the derivative of the co-motion functions.

        Parameters
        ----------
        Ne : (N_points,) array
            Cumulant of the electron density (at the position of electron 1).
        f : (N_points, Nel) array
            Co-motion functions if already computed. Default is no.

        Returns
        -------
        f_deriv : (N_points, Nel) array
            Derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_functions_Ne(Ne)
        f_deriv = jnp.empty((*Ne.shape, self.Nel)).at[:, 0].set(self.invNe_deriv(Ne))
        Nmax = (self.Nel + self.Nel % 2) // 2
        k = jnp.arange(1, Nmax)
        f_deriv = f_deriv.at[:, 1:-1:2].set(
            -jnp.sign(2 * k[None, :] - Ne[:, None])
            / (4 * jnp.pi * f[:, 1:-1:2] ** 2 * self.rho(f[:, 1:-1:2]))
        )
        f_deriv = f_deriv.at[:, 2::2].set(
            jnp.sign(self.Nel - 2 * k[None, :] - Ne[:, None])
            / (4 * jnp.pi * f[:, 2::2] ** 2 * self.rho(f[:, 2::2]))
        )
        if self.Nel % 2 == 0:
            f_deriv = f_deriv.at[:, -1].set(-1 / (4 * jnp.pi * f[:, -1] ** 2 * self.rho(f[:, -1])))
        return f_deriv

    def co_motion_functions_deriv2_Ne(
        self,
        Ne: jnp.ndarray,
        f: Optional[jnp.ndarray] = None,
        f_deriv: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute the second derivative of the co-motion functions.

        Parameters
        ----------
        Ne : (N_grid,) array
            Cumulant of the electron density (at the position of electron 1).
        f : (N_grid, Nel) array
            Co-motion functions if already computed. Default is no.
        f_deriv : (N_grid, Nel) array
            Derivative of the co-motion functions if already computed. Default is no.

        Returns
        -------
        f_deriv2 : (N_grid, Nel) array
            Second derivative of the co-motion functions.
        """
        if f is None:
            f = self.co_motion_functions_Ne(Ne)
        if f_deriv is None:
            f_deriv = self.co_motion_functions_deriv_Ne(Ne, f)
        return (-1) ** jnp.arange(self.Nel)[None, :] * (
            -2 * f_deriv / (4 * jnp.pi * f**3 * self.rho(f))
            - f_deriv * self.rho_deriv(f) / (4 * jnp.pi * f**2 * self.rho(f) ** 2)
        )

    def vH(self, r: jnp.ndarray):
        """
        Compute the Hartree potential.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        vH : (N_points,) array
            Hartree potential.
        """
        return self.Ne(r) / r + self.coNe(r)

    def vH_deriv(self, r: jnp.ndarray):
        """
        Compute the derivative of the Hartree potential.

        Parameters
        ----------
        r : (N_points,) array
            Radial coordinate.

        Returns
        -------
        vH_deriv : (N_points,) array
            Derivative of the Hartree potential.
        """
        return -self.Ne(r) / r**2


class NumericalInvNeDensity(Density, metaclass=abc.ABCMeta):
    """
    Baseclass for a density in which the inverse of the cumulant
    is computed numerically through a root solver.

    Additional attributes:
    ----------
    N_points : int
        Number of points in the grid used to compute the inverse of the cumulant.
    cumulant_threshold : float
        Threshold for the pre-inversion of the cumulant.
        That is, we expand our grid until the point r where Ne(r) = Nel - threshold.
    invNe_tol : float
        Tolerance for the root solver.

    Additional methods:
    ----------
    build_grid_guess :
        Find the point r_end, where Ne(r) = Nel - threshold.
        Build an evenly-spaced grid of N_points radial coordinates.
        Calculate the corresponding value of the cumulant Ne(r).
    newton_step :
        Perform a single newton iteration.
    newton_tol :
        Check if the error |Ne(r)- n| is below tol. Return False if yes, return True if not.
    newton :
        Perform newton iterations until the error is below tol.

    """

    __doc__ = Density.__doc__ + __doc__

    def __init__(
        self,
        Nel: int = 0,
        cumulant_threshold: float = 1e-6,
        N_points_Ne: int = 10000,
        invNe_tol: float = 1e-12,
    ):
        super().__init__(Nel=Nel)
        self.N_points_Ne = N_points_Ne
        self.cumulant_threshold = cumulant_threshold
        self.invNe_tol = invNe_tol

    def __post_init__(self):
        """
        Build a grid of radial coordinates and calculate the corresponding value
        of the cumulant Ne(r)
        """
        self.r_grid, self.Ne_grid = self.build_grid_guess()
        self.add_jvps()
        self.add_jit()
        super().__post_init__()

    def build_grid_guess(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build a grid of radial coordinates
        and calculate the corresponding value of the cumulant Ne(r)

        Returns
        -------
        r_grid : (N_points,) array
            Grid of radial coordinates.
        Ne_grid : (N_points,) array
            Cumulant evaluated at r_grid.
        """
        r_end = self.newton(1.0, self.Nel - self.cumulant_threshold)
        r_grid = jnp.linspace(0.0, r_end, self.N_points_Ne)
        Ne_grid = self.Ne(r_grid)
        return r_grid, Ne_grid

    def encode(self) -> Dict[str, Union[int, float, str]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "cumulant_threshold": self.cumulant_threshold,
                "N_points_Ne": self.N_points_Ne,
                "invNe_tol": self.invNe_tol,
            }
        )
        return encode_dict

    def newton_step(self, n: float, carry: Tuple[float, float]) -> Tuple[float, float]:
        """Perform a single newton step.

        Parameters
        ----------
        carry : Tuple[float, float]
            r : float
                Current value of the root.
            error : float
                Current error.

        Returns
        -------
        carry : Tuple[float, float]
            As above, but with the updated values.
        """
        r, _ = carry
        new_r = r - (self.Ne(r) - n) / self.Ne_deriv(r)
        error = jnp.abs(self.Ne(new_r) - n)
        return (new_r, error)

    def newton_tol(self, carry: Tuple[float, float]) -> bool:
        """Check whether the error is above the tol

        Parameters
        ----------
        carry : Tuple[float, float]
            r : float
                Current value of the root.
            error : float
                Current error.

        Returns
        -------
        bool
            True if the error is above the tol, else False.
        """
        _, error = carry
        return error > self.invNe_tol

    def newton(self, r0: float, n: float) -> float:
        """
        Perform a newton iteration to find the root of Ne(r) - n = 0.

        Parameters
        ----------
        r0 : float
            Initial guess for the root.
        n : float
            Value of the cumulant at the position of the electron.

        Returns
        -------
        r : float
            Root of Ne(r) - n = 0.
        """
        error0 = jnp.abs(self.Ne(r0) - n)
        r, _ = lax.while_loop(self.newton_tol, partial(self.newton_step, n), (r0, error0))
        return r

    def invNe(self, n: jnp.ndarray) -> jnp.ndarray:
        r0 = jnp.interp(n.reshape(-1), self.Ne_grid, self.r_grid)
        r = vmap(self.newton, in_axes=(0, 0))(r0, n.reshape(-1))
        return r.reshape(n.shape)
