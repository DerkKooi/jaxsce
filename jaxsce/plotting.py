"""Plotting"""

from typing import Dict, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import interp1d

# import matplotlib.pylab as pylab
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use(["seaborn-v0_8-paper", "seaborn-v0_8-dark-palette"])
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("font", family="sans-serif")
plt.rc("lines", markersize=8)


def scatter(
    x: jnp.ndarray,
    y: jnp.ndarray,
    title: str = "Scatter plot",
    xlabel: str = "x",
    ylabel: str = "y",
) -> Tuple[Figure, plt.Axes]:
    r"""Scatter plot

    Parameters
    ----------
    x : jnp.ndarray
        x-values
    y : jnp.ndarray
        y-values
    title : str, optional
        Title of the plot, by default "Scatter plot"
    xlabel : str, optional
        Label of the x-axis, by default "x"
    ylabel : str, optional
        Label of the y-axis, by default "y"

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def compare_Vee(
    Vee_A: jnp.ndarray,
    Vee_B: jnp.ndarray,
    title: str = "Energy of the initial and optimized angles",
    xlabel: str = r"$V_\mathrm{ee}$ start",
    ylabel: str = r"$V_\mathrm{ee}$ optimized",
) -> Tuple[Figure, plt.Axes]:
    r"""Compare the energy of the initial and optimized angles

    Parameters
    ----------
    Vee_A : jnp.ndarray
        Energy of the initial angles
    Vee_B : jnp.ndarray
        Energy of the optimized angles
    title : str, optional
        Title of the plot, by default "Energy of the initial and optimized angles"
    xlabel : str, optional
        Label of the x-axis, by default "$V_\mathrm{ee}$ start"
    ylabel : str, optional
        Label of the y-axis, by default "$V_\mathrm{ee}$ optimized"

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes of the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(Vee_A, Vee_B, "o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    min_Vee = jnp.min(jnp.concatenate((Vee_A, Vee_B)))
    max_Vee = jnp.max(jnp.concatenate((Vee_A, Vee_B)))
    ax.plot([min_Vee, max_Vee], [min_Vee, max_Vee], color="k", linestyle="-", linewidth=2)
    ax.set_xlim([jnp.min(Vee_A), jnp.max(Vee_A)])
    ax.set_ylim([jnp.min(Vee_B), jnp.max(Vee_B)])
    return fig, ax


def histogram_Vee(
    Vee_dict: Dict[str, jnp.ndarray],
    title: str = "Distribution of the energy after optimization",
    xlabel: str = r"$V_\mathrm{ee}$",
    ylabel: str = r"P($V_\mathrm{ee}$)",
) -> Tuple[Figure, plt.Axes]:
    r"""Plot the distribution of the energy after optimization

    Parameters
    ----------
    Vee_dict : Dict[str, jnp.ndarray]
        Dictionary with the energy of the optimized angles
    title : str, optional
        Title of the plot, by default "Distribution of the energy after optimization"
    xlabel : str, optional
        Label of the x-axis, by default "$V_\mathrm{ee}$"
    ylabel : str, optional
        Label of the y-axis, by default "P($V_\mathrm{ee}$)"

    Returns
    -------
    Tuple[Figure, plt.Axes]
        Figure and axes of the plot"""
    colors = ["r", "g", "b", "c", "m", "y", "k"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    for i, (key, Vee) in enumerate(Vee_dict.items()):
        _, _, _ = ax.hist(Vee, bins=50, density=True, facecolor=colors[i], alpha=0.75, label=key)
        ax.axvline(jnp.min(Vee), color=colors[i])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax


def gridplot_Vee(
    r: jnp.ndarray,
    Vee_dict: Dict[str, jnp.ndarray],
    title: str = r"V_\mathrm{ee} as a function of the position on the grid",
    xlabel: str = r"$r$",
    ylabel: str = r"$V_\mathrm{ee}$",
    N_interp: int = 10000,
) -> Tuple[Figure, plt.Axes]:
    r"""Plot the energy as a function of the position on the grid

    Parameters
    ----------
    r : jnp.ndarray
        Position on the grid
    Vee_dict : Dict[str, jnp.ndarray]
        Dictionary with the energy of the optimized angles
    title : str, optional
        Title of the plot, by default "Vee as a function of the position on the grid"
    xlabel : str, optional
        Label of the x-axis, by default r"$r$"
    ylabel : str, optional
        Label of the y-axis, by default r"$V_\mathrm{ee}$"
    N_interp : int, optional
        Number of points to interpolate Vee, by default 10000
    """
    r_interp = jnp.linspace(jnp.min(r), jnp.max(r), N_interp)
    colors = ["r", "g", "b", "c", "m", "y", "k"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    for i, (key, Vee) in enumerate(Vee_dict.items()):
        ax.scatter(r, Vee, c=colors[i], label=key)
        interp1d_Vee = interp1d(r, Vee, kind="cubic")
        ax.plot(r_interp, interp1d_Vee(r_interp), color=colors[i + 1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax
