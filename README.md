# jaxsce: jax accelerated calculations for Strictly-Correlated Electrons

`jaxsce` performs calculations for the Strictly-Correlated Electrons (SCE) reference system. This is the code used in "Large-Z atoms in the strong-interaction limit of DFT: Implications for gradient expansions and for the Lieb-Oxford bound", Timothy J. Daas, Derk P. Kooi, Tarik Benyahia, Michael Seidl, Paola Gori-Giorgi, [arXiv:2211.07512](https://arxiv.org/abs/2211.07512)


## Features
`jaxsce` contains
- `jaxsce.coordinates_3d`, which supports several spherical coordinates systems with the Coulomb interaction.
- `jaxsce.densities`, which features the base classes `Density` and derived class `NumericalInvNeDensity` in `jaxsce.densities.classes`. `NumericalInvNeDensity` implements a Newton method to invert the cumulant. It also features a collection of specific densities implemented:
1. `jaxsce.densities.sqrt_r`: $\sqrt{r}$ times an exponential density with analytical cumulants, inverse cumulants and co-motion functions. Derives from `Density`.
2. `jaxsce.densities.pyscf`: Hartree-Fock and CCSD densities from `pyscf` with cumulants calculated directly from `pyscf` integrals. Inverse cumulants and therefore the co-motion functions are obtained via numerical inversion of the cumulant using the Newton method. Derives from `NumericalInvNeDensity`.
3. `jaxsce.densities.bohr_atom`: Bohr atom densities with cumulants obtained directly from `sympy` via `sympy2jax`. Inverse cumulants and therefore the co-motion functions are obtained via numerical inversion of the cumulant using the Newton method. Derives from `NumericalInvNeDensity`.
4. `jaxsce.densities.test`: several test densities (that is Exponential and a Gaussian divided by $r$) that serve to test the base classes and derive from `NumericalInvNeDensity`.
- `jaxsce.optimize`, which features the class `AngularOptimization` to optimize the optimal angles for 3D spherical symmetry at a series of points. It produces $V_\mathrm{ee}^\mathrm{SCE}(r)$ on the grid points as well as its first and second derivative and stores them in a serializable `AngularOptimizationResult` object. Several features are implemented to obtain accurate minima for the highly non-convex angular optimization problem. In particular, we can first obtain starting points using random angles for several of the points and then "sweep" along the grid to obtain increasingly lower minima. Specifically for two-electron systems, in which there is no angular optimization problem, there is the `TwoElectron` class, which also creates an `AngularOptimizationResult` object containing the relevant quantities on the grid. The `AngularOptimizationResult` object is contains, aside from the results of the optimization, all the coordinates, density and optimizer parameters. This allows for easy storage of the results, and records all parameters involved in the calculation.
- `jaxsce.integrate`, which features the class `AngularIntegration`, which integrates $V_\mathrm{ee}^\mathrm{SCE}(r)$ from an `AngularOptimizationResult` object on the grid to obtain $V_\mathrm{ee}^\mathrm{SCE}[\rho]$. Several features are implemented to obtain an accurate integration. In particular, a variety of integration methods can be selected from `scipy` and integration can be done using Bernstein polynomials including the derivatives of the integrand in the interpolation. Finally, Richardson extrapolation is used to obtain a converged result.

## Data
The raw data used in the paper can be found in the subdirectory `data` with subdirectories for different systems. The integrals and derived quantities $\Lambda$ and $B$ are stored in `1025_integrals.json`. The 1025 in the filename refers to the number of gridpoints. To load the raw results of the optimization, including all the parameters used one can use for example the following code:
```from jaxsce.optimize import AngularOptimizationResult

res = AngularOptimizationResult.load("data/sqrt_r/10/1025")
```
One can access the `AngularOptimization` object from `AngularOptimizationResult` as `res.opt`. This then provides access also to the density via `res.opt.density` and the coordinates via `res.opt.coordinates`.

## Dependencies
The core of `jaxsce` depends on the following packages:
- `numpy`
- `scipy`
- `jax`
- `jaxopt`
- `dataclasses-json`

For the neutral atom densities [jaxsce] additionally depends on:
- `pyscf`
- `basis-set-exchange`

For the Bohr atom densities [jaxsce] additionally depends on:
- `sympy`
- `sympy2jax`
- `tensorflow`
- `tensorflow-probability`

## Setup:
- Clone repository
- cd jaxse
- pip install .

or

- pip install git+https://github.com/DerkKooi/jaxsce

## Usage
End-to-end use is demonstrated in `run_optimization.py`, with a description of all the command line arguments accessible via `python run_optimization.py --help`. It uses an equidistant grid between the points where the density integrates to 1 ($a_1$) and 2 ($a_2$), which is multiplied by $N$ to obtain the final result. Briefly, the density and coordinates objects are initialized, the optimizer is intialized and then ran producing an `AngularOptimizationResult` object that contains the results and all the parameters that went into the calculation. This can then be saved and easily loaded again. 

The `AngularOptimizationResult` object is then passed to the integration routine, which produces the final result. Integration with extrapolation is automatically carried out with a variety of methods, which can be compared to assess convergence. In `run_optimization.py` the Anderson extrapolated integral obtained from the Bernstein polynomial interpolation of second order is used. The convergence of the integral is improved by subtracting a contribution of which the derivative diverges at $a_2$. Using Anderson extrapolation, however, all methods with and without this subtraction converge to roughly the same result.

The obtained $V_\mathrm{ee}^\mathrm{SCE}[\rho]$ is combined with the LDA and GEA integrals to produce the quantities studied in the paper. All integrals are stored together in a single json file.

## Caveats (!)
1. Although the method should work for both even and odd numbers of electrons, it is completely untested for odd numbers of eletrons.
2. The default settings for `run_optimization.py` were used with an A100, for other systems the number of grid points and the number of random starting points may need to be adjusted.
3. The second derivative of the co-motion function has the wrong sign outside of `a1` to `a2`. 

## License
MIT License

Copyright (c) 2023 Derk P. Kooi, Kimberly J. Daas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
