import os

import jax.numpy as jnp
import pytest
from jax.config import config

from jaxsce import densities

config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("sqrt_r", {"Nel": 10}),
        ("bohr_atom", {"Nel": 10}),
        ("sqrt_r", {"Nel": 18}),
        ("bohr_atom", {"Nel": 18}),
        (
            "hartree-fock",
            {
                "atom": "Ne",
                "charge": 0,
                "spin": 0,
                "basis": "cc-pvdz",
                "chkfile_name": "some_chkfile",
            },
        ),
        (
            "ccsd",
            {
                "atom": "Ne",
                "charge": 0,
                "spin": 0,
                "basis": "cc-pvdz",
                "dm_file_name": "some_file_dm",
                "chkfile_name": "some_chkfile",
            },
        ),
    ],
)
def test_density(name: str, kwargs: dict):
    # Intitialize density
    density = densities.get_density(name, **kwargs)

    # Check that density has all the required attributes
    assert density.Nel
    assert jnp.all(density.a >= 0.0)
    assert density.a.shape == (density.Nel,)
    assert density.LDA_int
    assert density.GEA_int

    r = density.a[1:]

    # Test positivity of density
    rho = density.rho(r)
    assert jnp.all(rho >= 0.0)

    # Test that cumulant integrates to the correct number of electrons
    Ne = density.Ne(r)
    assert jnp.allclose(Ne, jnp.arange(1, density.Nel))

    # Test that the derivative of the cumulant is correct
    Ne_deriv = density.Ne_deriv(r)
    assert jnp.allclose(Ne_deriv, 4 * jnp.pi * r**2 * rho)

    # Test if the second derivative works
    density.Ne_deriv2(r)

    # Test that the co-Cumulant is positive
    coNe = density.coNe(r)
    assert jnp.all(coNe >= 0.0)

    # Test that the derivative of the co-Cumulant works
    density.coNe_deriv(r)

    # Test that the inverse of the cumulant is correct
    invNe = density.invNe(Ne)
    assert jnp.allclose(invNe, r)

    # Test that the derivative of the inverse of the cumulant is correct
    invNe_deriv = density.invNe_deriv(Ne, invNe)
    assert jnp.allclose(invNe_deriv, 1.0 / Ne_deriv)

    # Test that the co-motion functions are positive
    co_motion_function = density.co_motion_function(r[1])
    assert jnp.all(co_motion_function >= 0.0)
    co_motion_functions = density.co_motion_functions(r)
    assert jnp.all(co_motion_functions >= 0.0)
    assert jnp.allclose(co_motion_function, co_motion_functions[1])

    # Test that the co-motion functions and its derivatives computed from Ne
    # are equal to the ones computed from r=a
    co_motion_function_Ne = density.co_motion_function_Ne(Ne[1])
    assert jnp.allclose(co_motion_function_Ne, co_motion_function)
    co_motion_functions_Ne = density.co_motion_functions_Ne(Ne)
    assert jnp.allclose(co_motion_functions_Ne, co_motion_functions)

    # We don't want to compute the derivatives at a, because the co-motions
    # there are singular
    r = jnp.array([0.5, 1.0, 1.5, 2.0])
    Ne = density.Ne(r)
    Ne_deriv = density.Ne_deriv(r)
    density.Ne_deriv2(r)
    co_motion_function_deriv = density.co_motion_function_deriv(r[1])
    co_motion_function_deriv_Ne = density.co_motion_function_deriv_Ne(Ne[1])
    assert jnp.allclose(co_motion_function_deriv, co_motion_function_deriv_Ne * Ne_deriv[1])
    co_motion_functions_deriv = density.co_motion_functions_deriv(r)
    co_motion_functions_deriv_Ne = density.co_motion_functions_deriv_Ne(Ne)
    # only test derivatives for a[1:] because co-motion functions are ill behaved at zero
    assert jnp.allclose(co_motion_functions_deriv, co_motion_functions_deriv_Ne * Ne_deriv[:, None])

    # TODO: Second derivatives need a sign fix!
    # co_motion_function_deriv2 = density.co_motion_function_deriv2(r[1])
    # co_motion_function_deriv2_Ne = density.co_motion_function_deriv2_Ne(Ne[1])
    # assert jnp.allclose(co_motion_function_deriv2,
    # co_motion_function_deriv2_Ne*Ne_deriv[1]**2+co_motion_function_deriv_Ne*Ne_deriv2[1])
    # co_motion_functions_deriv2 = density.co_motion_functions_deriv2(r)
    # co_motion_functions_deriv2_Ne = density.co_motion_functions_deriv2_Ne(Ne)
    # assert jnp.allclose(co_motion_functions_deriv2[:],
    # co_motion_functions_deriv2_Ne[:]*Ne_deriv[:, None]**2
    # +co_motion_functions_deriv_Ne*Ne_deriv2[:, None])

    # Test that vH is positive
    vH = density.vH(r)
    assert jnp.all(vH >= 0.0)

    # Test that the derivative of vH works
    density.vH_deriv(r)

    # Delete the chkfile and dm_file if they exist
    if "chkfile_name" in kwargs:
        os.remove(kwargs["chkfile_name"])
    if "dm_file_name" in kwargs:
        os.remove(kwargs["dm_file_name"] + ".npy")
