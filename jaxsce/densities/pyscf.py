"""Densities from PySCF."""
import abc
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Dict, Tuple, Union

import basis_set_exchange as bse
import numpy as np
import scipy
from pyscf import cc, df, dft, gto, scf
from pyscf.data.elements import NUC
from pyscf.gto.basis import BasisNotFoundError

from .classes import Density


def to_3D_vector(r: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert a 1D array of radial coordinates to cartesian coordinates.

    Parameters
    ----------
    r : (N_points,) array
        1D array of n radial coordinates.

    Returns
    -------
    xyz : (N_points, 3) array
        3D array of n cartesian coordinates.
    """
    if isinstance(r, float):
        return np.array([[r, 0, 0]])
    return np.concatenate([r[..., None], np.zeros((*r.shape, 2))], axis=-1)


class PyscfDensity(Density, metaclass=abc.ABCMeta):
    """
    Base class for densities from PySCF.

    Additional attributes:
    ---------------------
    atom : str
        Atom symbol (and coordinates if specified, but unnecessary).
    charge : int
        Charge of the atom.
    spin : int
        Spin of the atom.
    N_points : int
        Number of radial points to use for inverting Ne and numerical integration.
    threshold : float
        Threshold for the root solver.
        That is, we expand our grid until the point r where Ne(r) = Nel - threshold.
    tol : float
        Tolerance for the root solver.
    trunc : float
        Truncation threshold on the density for integration.
    maxiter : int
        Maximum number of iterations for the root solver.
    basis : str
        Basis set to use for PySCF.
    method : str
        Root solving method to use, either newton or halley. Default is newton.
    mol : pyscf.gto.Mole
        PySCF molecule object.
    symb : str
        Symbol of the atom. Equal to self.atom unless atom contains coordinates.
    """

    __doc__ = Density.__doc__ + __doc__

    def __init__(
        self,
        atom: str = "",
        charge: int = 0,
        spin: int = 0,
        N_points_Ne: int = 10000,
        N_int: int = 100000,
        cumulant_threshold: float = 1e-10,
        invNe_tol: float = 1e-12,
        rho_trunc: float = 1e-10,
        maxiter: int = 100,
        basis: str = "sto-3g",
        method: str = "newton",
    ):
        if atom == "":
            raise ValueError("atom must be specified")
        self.atom = atom[:2].strip()
        self.charge = charge
        self.spin = spin
        self.N_points_Ne = N_points_Ne
        self.N_int = N_int
        self.cumulant_threshold = cumulant_threshold
        self.invNe_tol = invNe_tol
        self.rho_trunc = rho_trunc
        self.maxiter = maxiter
        self.basis = basis
        self.method = method

        # Create PySCF molecule object, get number of electrons and atom symbol
        try:
            self.mol = gto.M(atom=atom, charge=charge, spin=spin, basis=basis, symmetry=True)
        except BasisNotFoundError:
            self.mol = gto.M(
                atom=atom,
                charge=charge,
                spin=spin,
                basis=bse.get_basis(basis, fmt="nwchem", elements=[NUC[atom]]),
                symmetry=True,
            )
        Nel = self.mol.nelectron

        super().__init__(Nel=Nel)

        # Set up functions for the root solver
        self.f = lambda r, n: self.Ne(r) - n
        self.fprime = lambda r, n: self.Ne_deriv(r)
        if method == "newton":
            self.fprime2 = None
        elif method == "halley":
            self.fprime2 = lambda r, n: self.Ne_deriv2(r)

        # Initialize the grid arrays
        self.r_grid: np.ndarray = None
        self.dr: np.ndarray = None
        self.Ne_grid: np.ndarray = None

    def __post_init__(self):
        self.r_grid, self.Ne_grid, self.dr = self.build_grid_guess()
        super().__post_init__()
        self.add_jvps()

        # Compute the integrals LDA_int and GEA_int
        r, dr = dft.radi.treutler(self.N_int)
        rho_r = self.rho(r)
        self.LDA_int = np.sum(dr * 4 * np.pi * r**2 * rho_r ** (4 / 3))
        non_zero = np.where(rho_r > self.rho_trunc)
        self.GEA_int = np.sum(
            dr[non_zero]
            * 4
            * np.pi
            * r[non_zero] ** 2
            * self.rho_deriv(r[non_zero]) ** 2
            / (rho_r[non_zero] ** (4 / 3))
        )

    def encode(self) -> Dict[str, Union[int, float, str]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "atom": self.atom,
                "charge": self.charge,
                "spin": self.spin,
                "N_points_Ne": self.N_points_Ne,
                "N_int": self.N_int,
                "cumulant_threshold": self.cumulant_threshold,
                "invNe_tol": self.invNe_tol,
                "rho_trunc": self.rho_trunc,
                "maxiter": self.maxiter,
                "basis": self.basis,
                "method": self.method,
            }
        )
        return encode_dict

    def build_grid_guess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a grid of radial coordinates
        and calculate the corresponding value of the cumulant Ne(r)

        Returns
        -------
        r_grid : (npoints,) array
            Grid of radial coordinates.
        Ne_grid : (npoints,) array
            Cumulant Ne evaluated at r_grid.
        dr : (npoints,) array
            Grid spacing. (4 * np.pi * r**2 * dr = weight)
        """
        r_grid, dr = dft.radi.treutler(self.N_points_Ne)
        Ne_grid = self.Ne(r_grid)
        return r_grid, Ne_grid, dr

    def invNe(self, n: np.ndarray) -> np.ndarray:
        # Create guess for the root solver on the grid
        r0 = np.interp(n, self.Ne_grid, self.r_grid)
        return scipy.optimize.newton(
            self.f,
            r0,
            fprime=self.fprime,
            args=(n,),
            tol=self.invNe_tol,
            maxiter=self.maxiter,
            fprime2=self.fprime2,
        )

    def rho(self, r: np.ndarray) -> np.ndarray:
        # Compute the value of the density in the AO basis
        ao_value = dft.numint.eval_ao(self.mol, to_3D_vector(r.reshape(-1)), deriv=0)
        if isinstance(r, (int, float)):
            # eval_rho returns an array instead of a float if the input is a float
            return dft.numint.eval_rho(self.mol, ao_value, self.dm, xctype="LDA")[0]
        return dft.numint.eval_rho(self.mol, ao_value, self.dm, xctype="LDA").reshape(r.shape)

    def rho_deriv(self, r: np.ndarray) -> np.ndarray:
        # Compute the value of the density and its gradient in the AO basis
        ao_value = dft.numint.eval_ao(self.mol, to_3D_vector(r.reshape(-1)), deriv=1)

        # Gradient to x = r is the second element of the array
        if isinstance(r, (int, float)):
            # eval_rho returns an array instead of a float if the input is a float
            return dft.numint.eval_rho(self.mol, ao_value, self.dm, xctype="GGA")[1][0]
        return dft.numint.eval_rho(self.mol, ao_value, self.dm, xctype="GGA")[1].reshape(r.shape)

    def rho_deriv2(self, r: np.ndarray) -> np.ndarray:
        # Compute the value of the density, its gradient and its laplacian in the AO basis
        ao_value = dft.numint.eval_ao(self.mol, to_3D_vector(r.reshape(-1)), deriv=2)
        rho_grad_lap = dft.numint.eval_rho(self.mol, ao_value, self.dm, xctype="MGGA")

        # Pyscf generates the laplacian as the fourth element of the array
        # laplacian rho(r) = d^2 rho(r)/dr^2 + 2 d rho(r)/dr /r,
        # so d^2 rho(r)/dr^2 = laplacian rho(r) - 2 d rho(r)/dr /r
        if isinstance(r, (int, float)):
            return rho_grad_lap[4][0] - 2 * rho_grad_lap[1][0] / r
        return rho_grad_lap[4].reshape(r.shape) - 2 * rho_grad_lap[1].reshape(r.shape) / r

    def vH(self, r: np.ndarray) -> np.ndarray:
        # We compute vH using a fakemol with very sharply peaked Gaussians
        if isinstance(r, (int, float)):
            fakemol = gto.fakemol_for_charges(to_3D_vector(r))
        else:
            # We need to flatten r to pass it to fakemol_for_charges
            fakemol = gto.fakemol_for_charges(to_3D_vector(r.reshape(-1)))

        # We compute the Coulomb integrals between the fakemol and the real molecule
        ints = df.incore.aux_e2(self.mol, fakemol, intor="int3c2e")
        ints = (ints + ints.transpose(1, 0, 2)) / 2
        if isinstance(r, (int, float)):
            # einsum returns an array instead of a float if the input is a float
            return np.einsum("ijp,ij->p", ints, self.dm)[0]
        # We reshape the result to match the shape of r
        return np.einsum("ijp,ij->p", ints, self.dm).reshape(r.shape)

    def vH_deriv(self, r: np.ndarray) -> np.ndarray:
        # We compute vH_deriv using a fakemol with very sharply peaked Gaussians
        if isinstance(r, (int, float)):
            fakemol = gto.fakemol_for_charges(to_3D_vector(r))
        else:
            # We need to flatten r to pass it to fakemol_for_charges
            fakemol = gto.fakemol_for_charges(to_3D_vector(r.reshape(-1)))

        # We compute the gradient of the Coulomb integrals between the fakemol and the real molecule
        # The derivative to x = r is th e first element of the array
        ints = df.incore.aux_e2(self.mol, fakemol, intor="int3c2e_ip1")[0]
        ints = ints + ints.transpose(1, 0, 2)
        if isinstance(r, (int, float)):
            # einsum returns an array instead of a float if the input is a float
            return np.einsum("ijp,ij->p", ints, self.dm)[0]
        # We reshape the result to match the shape of r
        return np.einsum("ijp,ij->p", ints, self.dm).reshape(r.shape)

    def Ne(self, r: np.ndarray) -> np.ndarray:
        # v_H(r) = Ne(r)/r + coNe(r)
        # The derivative of v_H(r) is then
        # 4 pi r rho(r) - Ne(r)/r^2 - 4 pi r rho(r)
        # = - Ne(r)/r^2
        return -(r**2) * self.vH_deriv(r)

    def coNe(self, r: np.ndarray) -> np.ndarray:
        # v_H(r) = Ne(r)/r + coNe(r)
        return self.vH(r) - self.Ne(r) / r


class HFDensity(PyscfDensity):
    """
    (Restricted) Hartree-Fock density class.

    Additional attributes:
    ----------------------
    scf: pyscf.scf.hf.RHF
        The pyscf object for the RHF calculation
    dm: numpy.ndarray
        The density matrix resulting from the RHF calculation
    """

    __doc__ = PyscfDensity.__doc__ + __doc__

    def __init__(self, chkfile_name: str = "", chkfile_dir="", **kwargs):
        super().__init__(**kwargs)
        self.name = "hartree-fock"

        # Setup the RHF calculation
        mf = scf.RHF(self.mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-11
        mf.max_cycle = 1000

        # Check if chkfile_name is given
        if chkfile_name == "":
            raise ValueError("chkfile_name must be given")
        self.chkfile_name = chkfile_name
        self.chkfile_dir = chkfile_dir
        chkfile = os.path.join(chkfile_dir, chkfile_name)

        if os.path.isfile(chkfile):
            # If chkfile exists, load it and update the scf object
            mf.chkfile = chkfile
            mf.update()
        else:
            # If chkfile does not exist, run the calculation and save the chkfile
            with NamedTemporaryFile() as tempchk:
                mf.chkfile = tempchk.name
                mf.kernel()
                shutil.copyfile(tempchk.name, chkfile)

        # Build the Hartree-Fock density matrix in the AO basis
        self.dm = mf.make_rdm1()
        self.U = np.trace(self.dm.dot(mf.get_j())) / 2

        # Now build the grid for Ne inversion, and calculate integrals
        self.__post_init__()

    def encode(self) -> Dict[str, Union[str, float, int]]:
        encode_dict = super().encode()
        encode_dict.update({"chkfile_name": self.chkfile_name, "chkfile_dir": self.chkfile_dir})
        return encode_dict


class CCSDDensity(PyscfDensity):
    """
    CCSD density class.

    Additional attributes:
    ----------------------
    scf: pyscf.scf.hf.RHF
        The pyscf object for the RHF calculation
    dm: numpy.ndarray
        The density matrix resulting from the CCSD calculation.
    """

    __doc__ = PyscfDensity.__doc__ + __doc__

    def __init__(
        self,
        dm_file_name: str = "",
        dm_file_dir: str = "",
        chkfile_name: str = "",
        chkfile_dir: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "ccsd"
        if dm_file_name == "":
            raise ValueError("dm_file_name must be given")
        self.dm_file_name = dm_file_name
        self.dm_file_dir = dm_file_dir
        dm_file = os.path.join(dm_file_dir, dm_file_name)

        if chkfile_name == "":
            raise ValueError("chkfile_name must be given")
        self.chkfile_name = chkfile_name
        self.chkfile_dir = chkfile_dir
        chkfile = os.path.join(chkfile_dir, chkfile_name)

        if os.path.isfile(dm_file):
            # Simply load the density matrix in AO basis if it exists
            self.dm = np.load(dm_file)
        else:
            # Run RHF calculation if necessary
            mf = scf.RHF(self.mol)
            mf.conv_tol = 1e-12
            mf.conv_tol_grad = 1e-11
            mf.max_cycle = 1000
            if os.path.exists(chkfile):
                # If chkfile exists, load it and update the scf object
                mf.chkfile = chkfile
                mf.update()
            else:
                with NamedTemporaryFile() as tempchk:
                    mf.chkfile = tempchk.name
                    mf.kernel()
                    shutil.copyfile(tempchk.name, chkfile)

            # Run CCSD calculation
            ccsd = cc.CCSD(mf)
            ccsd.kernel()
            self.dm = ccsd.make_rdm1()

            # Transform the density matrix to the AO basis
            self.dm = mf.mo_coeff.dot(self.dm.dot(mf.mo_coeff.T))
            np.save(dm_file, self.dm)

        eris = self.mol.intor("int2e")
        self.U = 1 / 2 * np.einsum("pqrs,pq,rs->", eris, self.dm, self.dm)

        # Now build the grid for Ne inversion, and calculate integrals
        self.__post_init__()

    def encode(self) -> Dict[str, Union[str, float, int]]:
        encode_dict = super().encode()
        encode_dict.update(
            {
                "dm_file_name": self.dm_file_name,
                "dm_file_dir": self.dm_file_dir,
                "chkfile_name": self.chkfile_name,
                "chkfile_dir": self.chkfile_dir,
            }
        )
        return encode_dict
