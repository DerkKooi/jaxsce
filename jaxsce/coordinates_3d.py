"""Module for the Coulomb energy of electrons in 3D."""

import abc
from typing import Dict, Tuple, Union

# import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jit, random


def get_coordinate_system(name: str, seed: int = 5) -> "SphericalCoordinateSystem":
    """Returns the coordinate system class corresponding to the given name.

    Parameters
    ----------
    name : str
        Name of the coordinate system class.
    Nel : int
        Number of electrons.
    seed : int, optional
        Seed for the random number generator, by default 5

    Returns
    -------
    SphericalCoordinateSystem
        Coordinate system class corresponding to the given name."""

    if name == "allangles":
        return AllAngles(seed)
    if name == "reduced":
        return ReducedAngles(seed)
    if name == "double_reduced":
        return DoubleReducedAngles(seed)
    raise ValueError("Coordinate system not recognized.")


def distmat_cartesian(coords: jnp.ndarray) -> jnp.ndarray:
    """Compute the distance matrix of all electrons in cartesian coordinates.

    Parameters
    ----------
    coords : (..., N, 3) array
        Cartesian coordinates of the N electrons.

    Returns
    -------
    distance_matrix : (..., N, N) array
        Distance matrix of all electrons.
    """
    triu_indices = jnp.triu_indices(coords.shape[-2], k=1)
    return jnp.sqrt(
        jnp.sum((coords[..., :, None, :] - coords[..., None, :, :]) ** 2, axis=-1)[
            ..., triu_indices[0], triu_indices[1]
        ]
    )


def Vee_cartesian(coords: jnp.ndarray):
    """Compute the Coulomb energy of all electrons in cartesian coordinates.

    Parameters
    ----------
    coords : (..., N, 3) array
        Cartesian coordinates of the N electrons.

    Returns
    -------
    Vee : float
        Coulomb energy of all electrons.
    """
    return jnp.sum(1 / distmat_cartesian(coords), axis=-1)


def spherical_to_cartesian(r: jnp.ndarray, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Convert spherical coordinates to cartesian coordinates.
    Parameters
    ----------
    r : array (..., N_grid,)
        Radius.
    theta : array (..., N_grid,)
        Polar angle.
    phi : array (..., N_grid,)
        Azimuthal angle.

    Returns
    -------
    coords : array (..., N_grid, 3)
        Cartesian coordinates.
    """
    return jnp.stack(
        (
            r * jnp.sin(theta) * jnp.cos(phi),
            r * jnp.sin(theta) * jnp.sin(phi),
            r * jnp.cos(theta),
        ),
        axis=-1,
    )


# TODO: Generalize to general coordinate system, not only spherical

jit_functions = [
    "Vee",
    "Vee_grad_angles",
    "Vee_grad_radials",
    "Vee_value_and_grad",
    "Vee_hessian_angles",
    "Vee_hessian_radials",
    "Vee_hessian_angles_radials",
    "crossdist",
    "crossdist_all",
]


class SphericalCoordinateSystem(metaclass=abc.ABCMeta):
    """Abstract class for the coordinate system of the Coulomb energy of electrons in 3D."""

    def __init__(self, seed: int):
        self.name: str = ""
        self.key = random.PRNGKey(seed)
        self.seed = seed
        for function in jit_functions:
            setattr(self, function, jit(getattr(self, function)))

    def encode(self) -> Dict[str, Union[int, str]]:
        """Encode the coordinate system.

        Returns
        -------
        Dict[str, Union[int, str]]
            Dictionary containing the name of the coordinate system and the seed.
        """
        return {"name": self.name, "seed": self.seed}

    @abc.abstractmethod
    def full_angles(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        """Compute the full angles from the reduced angles.

        Parameters
        ----------
        r : (N_el, ) array
            Radial coordinate of the electrons.
        angles : (x, ) array
            Reduced angles.


        Returns
        -------
        theta : (N_el,) array
            Full theta angles.
        phi : (N_el,) array
            Full phi angles
        """

    def cartesian_coordinates(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        """Compute the cartesian coordinates from the reduced angles.

        Parameters
        ----------
        r : (..., N_el, ) array
            Radial coordinate of the electrons.
        angles : (..., x, ) array
            Reduced angles.


        Returns
        -------
        cartesian_coordinates : (..., N_el, 3) array
            Cartesian coordinates.
        """
        theta, phi = self.full_angles(r, angles)
        return spherical_to_cartesian(r, theta, phi)

    def center_of_charge(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        """Compute the center of charge from the angles.

        Parameters
        ----------
        r : (..., N_el, ) array
            Radial coordinate of the electrons.
        angles : (..., x, ) array
            Angles.


        Returns
        -------
        center_of_charge : (..., 3) array
            Center of charge.
        """
        return jnp.sum(self.cartesian_coordinates(r, angles), axis=-2)

    def center_of_charge_norm(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        """Compute the norm of the center of charge from the angles.

        Parameters
        ----------
        r : (..., N_el, ) array
            Radial coordinate of the electrons.
        angles : (..., x, ) array
            Angles.


        Returns
        -------
        center_of_charge_norm : (...,) array
            Norm of the center of charge.
        """
        return jnp.linalg.norm(self.center_of_charge(r, angles), axis=-1)

    def dipole_moment(
        self, r: jnp.ndarray, angles: jnp.ndarray, center_of_charge: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the dipole moment from the reduced angles.

        Parameters
        ----------
        r : (..., N_el, ) array
            Radial coordinate of the electrons.
        angles : (..., x, ) array
            Reduced angles.
        center_of_charge : (..., 3) array
            Center of charge.


        Returns
        -------
        dipole_moment : (..., 3) array
            Dipole moment.
        """
        return jnp.sum(self.cartesian_coordinates(r, angles), axis=-2) - center_of_charge

    def quadrupole_moment(
        self, r: jnp.ndarray, angles: jnp.ndarray, center_of_charge: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the quadrupole moment from the reduced angles.

        Parameters
        ----------
        r : (..., N_el, ) array
            Radial coordinate of the electrons.
        angles : (..., x, ) array
            Reduced angles.
        center_of_charge : (..., 3) array
            Center of charge.


        Returns
        -------
        quadrupole_moment : (..., 3, 3) array
            Quadrupole moment.
        """
        coords = self.cartesian_coordinates(r, angles) - center_of_charge[..., None, :]
        return 3 * coords.T.dot(coords) - jnp.diag(jnp.sum(coords**2, axis=-1))

    def Vee(
        self,
        angles: jnp.ndarray,
        r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the Coulomb energy of all electrons in cartesian coordinates.

        Parameters
        ----------
        angles : (..., x, ) array
            Angular coordinates of the electrons.
        r : (..., N_el, ) array
            Radial coordinate of the electrons.


        Returns
        -------
        Vee : float or array (..., )
            Coulomb energy of all electrons.
        """
        return Vee_cartesian(self.cartesian_coordinates(r, angles))

    def Vee_grad_angles(self, angles: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient of the Coulomb energy of all electrons to the angles.


        Parameters
        ----------
        angles : (..., x,) array
            Angular coordinates of the N-1 electrons.
        r : (..., N,) array
            Radial coordinate of the N electrons.

        Returns
        -------
        Vee_grad_angles : (..., x,) array
            Gradient of the Coulomb energy of all electrons to the angles.
        """
        return grad(self.Vee)(angles, r)

    def Vee_grad_radials(self, angles: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient of the Coulomb energy of all electrons to the radials.

        Parameters
        ----------
        angles : (..., x,) array
            Angular coordinates of the N-1 electrons.
        r : (..., N,) array
            Radial coordinate of the N electrons.

        Returns
        -------
        Vee_grad_radials : (..., x,) array
            Gradient of the Coulomb energy of all electrons to the radials.
        """
        return grad(self.Vee, argnums=1)(angles, r)

    def Vee_value_and_grad(
        self, angles: jnp.ndarray, r: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the Coulomb energy of all electrons in spherical coordinates and its gradient.

        Electron 1 is fixed on the z-axis. Electron 2 is fixed in the xz-plane.
        The remaining electrons can have any position.

        Parameters
        ----------
        angles : array
            Angular coordinates of the electrons.
        r : (..., N, N_points) array
            Radial coordinate of the N electrons.

        Returns
        -------
        Vee : float or array (...,)
            Coulomb energy of all electrons.
        Vee_grad_angles :  array
            Gradient of the Coulomb energy of all electrons to the angles.
        """
        return self.Vee(angles, r), self.Vee_grad_angles(angles, r)

    def Vee_hessian_angles(self, angles: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        """Compute the hessian of the Coulomb energy of all electrons.

        Parameters
        ----------
        angles : (..., x,) array
            Angular coordinates of the N-1 electrons.
        r : (..., N,) array
            Radial coordinate of the N electrons.

        Returns
        -------
        Vee_hessian : (x, x) array
            Hessian of the Coulomb energy of all electrons to the angles.
        """
        return jacfwd(grad(self.Vee))(angles, r)

    def Vee_hessian_radials(self, angles: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        """Compute the hessian of the Coulomb energy of all electrons to the radials.

        Parameters
        ----------
        angles : (..., x,) array
            Angular coordinates of the N-1 electrons.
        r : (..., N,) array
            Radial coordinate of the N electrons.

        Returns
        -------
        Vee_hessian : (x, x) array
            Hessian of the Coulomb energy of all electrons to the radials.
        """
        return jacfwd(grad(self.Vee, argnums=1), argnums=1)(angles, r)

    def Vee_hessian_angles_radials(self, angles: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        """Compute the hessian of the Coulomb energy of all electrons to the angles and radials.

        Parameters
        ----------
        angles : (..., x,) array
            Angular coordinates of the N-1 electrons.
        r : (..., N,) array
            Radial coordinate of the N electrons.

        Returns
        -------
        Vee_hessian : (x, x) array
            Hessian of the Coulomb energy of all electrons to the angles and radials.
        """
        return jacfwd(grad(self.Vee), argnums=1)(angles, r)

    @abc.abstractmethod
    def zero_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        """Generate zero angles for multiple points.

        Returns
        -------
        angles : (*args, Nel, x,)  array
            Zero angles.
        """

    @abc.abstractmethod
    def empty_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        """Generate zero angles for multiple points.

        Returns
        -------
        angles : (*args, Nel, x,)  array
            Zero angles.
        """

    @abc.abstractmethod
    def random_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        """Generate random angles for N_attempts.

        Parameters
        ----------
        *args : int
            First dimensions of the array.

        Returns
        -------
        angles : (*args, x)  array
            Random angles.
        """

    def split_key(self):
        """Split the random key."""
        _, self.key = random.split(self.key)

    def crossdist(
        self, rA: jnp.ndarray, anglesA: jnp.ndarray, rB: jnp.ndarray, anglesB: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the distance between two sets of coordinates.

        Note: does not use symmetry

        Parameters
        ----------
        rA : (N_el,)  array
            Radial coordinates of set A.
        anglesA : ( x,)  array
            Angles of set A.
        rB : ( N_el,)  array
            Radial coordinates of set B.
        anglesB : (x,)  array
            Random angles.

        Returns
        -------
        crossdist : jnp.ndarray
            Distance between the two sets of angles.
        """
        return jnp.sum(
            jnp.linalg.norm(
                self.cartesian_coordinates(rA, anglesA) - self.cartesian_coordinates(rB, anglesB),
                axis=-1,
            ),
            axis=-1,
        )

    def crossdist_all(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        """Compute the distance between all pairs of coordinates.

        Parameters
        ----------
        r : (..., N_el)  array
            Radial coordinates of all attempts.
        angles : (..., N_attempts, x)  array
            Angles of all points.

        Returns
        -------
        crossdist_all : (..., N_attempts, N_attempts)  array
            Distance between all pairs of points.
        """
        coords = self.cartesian_coordinates(
            r[..., None, :], angles
        )  # shape (..., N_attempts, N_el, 3)
        return jnp.sum(
            jnp.linalg.norm(coords[..., None, :, :, :] - coords[..., :, None, :, :], axis=-1),
            axis=-1,
        )


class AllAngles(SphericalCoordinateSystem):
    """Coordinate system where all angles are used, not making use of spherical symmetry."""

    def __init__(self, seed):
        super().__init__(seed)
        self.name = "all angles"

    def full_angles(self, r: jnp.ndarray, angles: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the full angles from the reduced angles.

        Parameters
        ----------
        r : (..., N_el, ) array
            Radial coordinate of the electrons.
        angles : (..., x, ) array
            Reduced angles.


        Returns
        -------
        theta : (..., N_el,) array
            Full theta angles.
        phi : (..., N_el,) array
            Full phi angles
        """
        return angles[..., 0], angles[..., 1]

    def random_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        theta = jnp.pi * random.uniform(self.key, (*args, Nel))
        phi = 2 * jnp.pi * random.uniform(self.key, (*args, Nel))

        # Zero first electron angles and second electron azimuthal angle
        theta[..., 0] = 0.0
        phi[..., 0] = 0.0
        phi[..., 1] = 0.0

        # Stack angles
        angles = jnp.concatenate((theta, phi), axis=-1)
        return angles

    def zero_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        return jnp.zeros((*args, Nel, 2))

    def empty_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        return jnp.empty((*args, Nel, 2))


class ReducedAngles(SphericalCoordinateSystem):
    """Coordinate system where the angles of electron 1,
     and the azimuthal angle of electron 2 are fixed.

    That is, we are making use of spherical symmetry."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.name = "reduced"

    def full_angles(self, r: jnp.ndarray, angles: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        theta = jnp.concatenate(
            (jnp.zeros((*angles.shape[:-1], 1)), angles[..., : r.shape[-1] - 1]), axis=-1
        )
        phi = jnp.concatenate(
            (jnp.zeros((*angles.shape[:-1], 2)), angles[..., r.shape[-1] - 1 :]), axis=-1
        )
        return theta, phi

    def random_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        theta = jnp.pi * random.uniform(self.key, (*args, (Nel - 1)))
        phi = 2 * jnp.pi * random.uniform(self.key, (*args, (Nel - 2)))

        # Stack angles
        angles = jnp.concatenate((theta, phi), axis=-1)
        return angles

    def zero_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        return jnp.zeros((*args, 2 * Nel - 3))

    def empty_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        return jnp.empty((*args, 2 * Nel - 3))

    def crossdist(
        self, rA: jnp.ndarray, anglesA: jnp.ndarray, rB: jnp.ndarray, anglesB: jnp.ndarray
    ) -> jnp.ndarray:
        coordsA = self.cartesian_coordinates(rA, anglesA)
        coordsB = self.cartesian_coordinates(rB, anglesB)
        return jnp.min(
            jnp.stack(
                (
                    jnp.sum(jnp.linalg.norm(coordsA - coordsB, axis=-1), axis=-1),
                    jnp.sum(
                        jnp.linalg.norm(
                            coordsA.at[..., 0].set(-coordsA[..., 0], axis=-1) - coordsB
                        ),
                        axis=-1,
                    ),
                ),
                axis=-1,
            ),
            axis=-1,
        )

    def crossdist_all(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        coords = self.cartesian_coordinates(
            r[..., None, :], angles
        )  # shape (..., N_attempts, N_el, 3)
        dist_direct = jnp.sum(
            jnp.linalg.norm(coords[..., None, :, :, :] - coords[..., :, None, :, :], axis=-1),
            axis=-1,
        )
        dist_mirror_yz = jnp.sum(
            jnp.linalg.norm(
                coords.at[..., 0].set(-coords[..., 0])[..., None, :, :, :]
                - coords[..., :, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        dist_mirror_xz = jnp.sum(
            jnp.linalg.norm(
                coords.at[..., 1].set(-coords[..., 1])[..., None, :, :, :]
                - coords[..., :, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        dist_mirror_yz_xz = jnp.sum(
            jnp.linalg.norm(
                coords.at[..., 0]
                .set(-coords[..., 0])
                .at[..., 1]
                .set(-coords[..., 1])[..., None, :, :, :]
                - coords[..., :, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        return jnp.min(
            jnp.stack((dist_direct, dist_mirror_yz, dist_mirror_xz, dist_mirror_yz_xz), axis=-1),
            axis=-1,
        )


class DoubleReducedAngles(SphericalCoordinateSystem):
    """Coordinate system where the angles of electron 1 and 2 are fixed to be zero.
    This can be done if one of the two has radials zero. Third electron has a fixed azimuthal angle.

    That is, we are making use of spherical symmetry twice."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.name = "double_reduced"

    def full_angles(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.concatenate(
            (jnp.zeros((*angles.shape[:-1], 2)), angles[..., : r.shape[-1] - 2]), axis=-1
        )
        phi = jnp.concatenate(
            (jnp.zeros((*angles.shape[:-1], 3)), angles[..., r.shape[-1] - 2 :]), axis=-1
        )
        return theta, phi

    def random_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        theta = jnp.pi * random.uniform(self.key, (*args, (Nel - 2)))
        phi = 2 * jnp.pi * random.uniform(self.key, (*args, (Nel - 3)))

        # Stack angles
        angles = jnp.concatenate((theta, phi), axis=-1)
        return angles

    def zero_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        return jnp.zeros((*args, 2 * Nel - 5))

    def empty_angles(self, Nel: int, *args: int) -> jnp.ndarray:
        return jnp.empty((*args, 2 * Nel - 5))

    def crossdist(
        self, rA: jnp.ndarray, anglesA: jnp.ndarray, rB: jnp.ndarray, anglesB: jnp.ndarray
    ) -> jnp.ndarray:
        coordsA = self.cartesian_coordinates(rA, anglesA)
        coordsB = self.cartesian_coordinates(rB, anglesB)
        return jnp.min(
            jnp.stack(
                (
                    jnp.sum(jnp.linalg.norm(coordsA - coordsB, axis=-1), axis=-1),
                    jnp.sum(
                        jnp.linalg.norm(
                            coordsA.at[..., 0].set(-coordsA[..., 0], axis=-1) - coordsB
                        ),
                        axis=-1,
                    ),
                ),
                axis=-1,
            ),
            axis=-1,
        )

    def crossdist_all(self, r: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        coords = self.cartesian_coordinates(
            r[..., None, :], angles
        )  # shape (..., N_attempts, N_el, 3)
        dist_direct = jnp.sum(
            jnp.linalg.norm(coords[..., None, :, :, :] - coords[..., :, None, :, :], axis=-1),
            axis=-1,
        )
        dist_mirror_yz = jnp.sum(
            jnp.linalg.norm(
                coords.at[..., 0].set(-coords[..., 0])[..., None, :, :, :]
                - coords[..., :, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        dist_mirror_xz = jnp.sum(
            jnp.linalg.norm(
                coords.at[..., 1].set(-coords[..., 1])[..., None, :, :, :]
                - coords[..., :, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        dist_mirror_yz_xz = jnp.sum(
            jnp.linalg.norm(
                coords.at[..., 0]
                .set(-coords[..., 0])
                .at[..., 1]
                .set(-coords[..., 1])[..., None, :, :, :]
                - coords[..., :, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        return jnp.min(
            jnp.stack((dist_direct, dist_mirror_yz, dist_mirror_xz, dist_mirror_yz_xz), axis=-1),
            axis=-1,
        )
