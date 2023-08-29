"""The densities module contains the classes corresponding to different electron densities.."""

from .bohr_atom import BohrAtom
from .pyscf import CCSDDensity, HFDensity
from .sqrt_r import SqrtR


def get_density(name: str = "", Nel: int = 0, **kwargs: dict):
    """
    Return the density class corresponding to the given name.

    Parameters
    ----------
    name : str
        Name of the density class.
    kwargs : dict
        Keyword arguments to pass to the density class.

    Returns
    -------
    Density
        Density class corresponding to the given name.
    """

    if name == "sqrt_r":
        return SqrtR(Nel=Nel, **kwargs)
    if name == "bohr_atom":
        return BohrAtom(Nel=Nel, **kwargs)
    if name == "hartree-fock":
        return HFDensity(**kwargs)
    if name == "ccsd":
        return CCSDDensity(**kwargs)

    raise ValueError(f"Unknown density {name}")
