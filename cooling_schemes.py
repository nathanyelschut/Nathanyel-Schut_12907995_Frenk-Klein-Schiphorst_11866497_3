"""Module containing several cooling scheme functions."""

from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np


def quadr_mult(iteration: int, initial_temp: int or float,
               alpha: float) -> float:
    """Quadratic additive cooling scheme.

    Args:
        iteration (int): step number of the Markov Chain
        initial_temp (int or float): starting temperature
        alpha (float): shape parameter of the cooling curve

    Returns:
        float: temperature at the given iteration
    """
    return initial_temp / (1 + alpha * iteration**2)


def exp_mult(iteration: int, initial_temp: int or float,
             alpha: float) -> float:
    """Exponential multiplicative cooling scheme.

    Args:
        iteration (int): step number of the Markov Chain
        initial_temp (int or float): starting temperature
        alpha (float): shape parameter of the cooling curve

    Returns:
        float: temperature at the given iteration
    """
    return initial_temp * alpha**iteration


def plot_cooling_scheme(cooling_scheme: FunctionType or list[FunctionType],
                        initial_temp: int or float, alphas:
                        float or list[float], iterations: int, title: str,
                        labels: list[str] = None, figsize: tuple = (6, 4.5),
                        zoom_window: bool = False, zoom_xlim: tuple = None):
    """Plots a cooling scheme for a specified value of alpha (or plots
    multiple cooling schemes each with their own alpha value if cooling
    schemes and alpha are passed as lists).

    Args:
        cooling_scheme (FunctionType or list): cooling schedule
            function which takes 3 arguments; the iteration, the
            initial temperature and an alpha value. Can also be passed
            as a list of cooling schemes
        initial_temp (int or float): initial temperature of the cooling
            scheme
        alphas (float or list): parameter for the cooling schedule. Can
            also be passed as a list of floats in which case the list
            should have the same length as the cooling scheme list
        iterations (int): number of iterations to plot
        title (str): plot title
        labels (list, optional): list of legend labels if cooling
            scheme is passed as a list. Defaults to None.
        figsize (tuple, optional): figure size (x, y). Defaults to
            (6, 4.5).
        zoom_window (bool, optional): whether or not to include a
            zoomed in window to show different limits. Defaults to
            False.
        zoom_xlim (tuple, optional): x limits of the zoomed in window.
            Defaults to None.
    """
    _, ax = plt.subplots(1, 1, figsize=figsize)
    if zoom_window:
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])

    iterations_l = list(range(iterations))

    if isinstance(cooling_scheme, (list, np.ndarray)):
        for i, scheme in enumerate(cooling_scheme):
            temperatures = [scheme(iteration, initial_temp, alphas[i])
                            for iteration in iterations_l]

            if labels is not None:
                ax.plot(range(iterations), temperatures, label=labels[i])

            else:
                ax.plot(range(iterations), temperatures)

            if zoom_window:
                axins.plot(range(iterations), temperatures)

    elif isinstance(cooling_scheme, FunctionType):
        temperatures = [cooling_scheme(iteration, initial_temp, alphas)
                        for iteration in iterations_l]

        ax.plot(range(iterations), temperatures)
        if zoom_window:
            axins.plot(range(iterations), temperatures)

    y_bounds = (-(0.01 * initial_temp), 1.01 * initial_temp)

    if zoom_window:
        axins.set(xbound=zoom_xlim, ybound=y_bounds)
        axins.grid(axis='y', alpha=0.5)

    ax.set_xlim(0, iterations)
    ax.set_ylim(y_bounds)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Temperature')

    ax.set_title(title)
    ax.grid(axis='y', alpha=0.5)

    if labels is not None:
        if zoom_window:
            ax.legend(loc='lower right')
        else:
            ax.legend()

    plt.show()
