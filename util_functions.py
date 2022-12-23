"""Module containing utility functions for doing experiments on
simulated annealing in the context of the travelling salesman
problem. The module includes file reading functions, plot functions
and data processing functions."""

# Authors: Nathanyel Schut and Frenk Klein Schiphorst
# Date: 19-12-2022

import itertools
import multiprocessing as mp
import re
from math import ceil, exp
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

import tsp_solver


def read_coordinates(file_name: str) -> dict:
    """Function for reading coordinates from a provided file.

    Args:
        file_name (str): path to the file containing the coordinates

    Returns:
        dict: contains the coordinates for each location in the problem
    """
    with open(file_name, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        lines = ''.join(lines)

        pattern = re.compile(r'(\d+)\s+(\d+)\s+(\d+)')
        result = pattern.findall(lines)

    coord_dict = {}
    for city, x_i, y_i in result:
        coord_dict[int(city)] = [int(x_i), int(y_i)]

    return coord_dict


def read_optimal_route(file_name: str) -> list:
    """Function for reading the configuration of the optimal route from
    a provided file.

    Args:
        file_name (str): path to the file containing the optimal
            configuration

    Returns:
        list: optimal configuration
    """
    with open(file_name, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        lines = ''.join(lines)

        pattern = re.compile(r'\n(\d+)')
        result = pattern.findall(lines)

        config = np.array(result, dtype=int)

    return config


def test_initial_temp(T_i: int or float, coord_dict: dict, N: int) -> tuple:
    """Function for finding the average acceptance probability of N
    random permutations (with increasing cost) for a given initial
    temperature.

    Args:
        T_i (int or float): initial temperature to test
        coord_dict (dict): coordinate dictionary, keys represent the
            city number, while values give the coordinate of that city
        N (int): number of permutations over which to take the average

    Returns:
        tuple: size 2 tuple containing the average acceptance
            probability and the standard deviation of the acceptance
            probabilities
    """
    # Initial configuration
    n_cities = len(coord_dict)
    config = np.arange(1, n_cities+1, 1)
    np.random.shuffle(config)
    cost = tsp_solver.cost_func(config, coord_dict)

    acceptance_probabilities = []

    # Test new configurations with a higher cost than the initial cost
    for _ in range(N):
        new_cost = 0

        while new_cost < cost:
            new_config = tsp_solver.permutate(config)
            new_cost = tsp_solver.cost_func(new_config, coord_dict)

        acceptance_probabilities.append(exp((cost-new_cost) / T_i))

    return acceptance_probabilities


def plot_configuration(config: list or np.ndarray, coord_dict: dict,
                       mark_first_last=False):
    """Plots a given configuration list based on the coordinates
    provided.

    Args:
        config (ArrayLike): configuration, given by a list or array of
            cities in order of their visit
        coord_dict (dict): coordinate dictionary, keys represent the
            city number, while values give the coordinate of that city
        mark_first_last (bool, optional): whether to indicate the first
            city (in green) and the last city (in red). Defaults to
            False.
    """
    # Order coordinates based on the configuration provided
    ordered_x = []
    ordered_y = []

    for city in config:
        ordered_x.append(coord_dict[city][0])
        ordered_y.append(coord_dict[city][1])

    ordered_x.append(ordered_x[0])
    ordered_y.append(ordered_y[0])

    cost = tsp_solver.cost_func(config, coord_dict)

    # Calculate plot bounds
    max_x = max(ordered_x)
    upper_x = ceil(max_x / 10) * 10

    if upper_x - max_x < 2:
        upper_x += 10

    max_y = max(ordered_y)
    upper_y = ceil(max_y / 10) * 10

    if upper_y - max_y < 2:
        upper_y += 10

    ratio = upper_x / upper_y

    # Plotting
    plt.figure(figsize=(6*ratio, 6))

    plt.plot(ordered_x, ordered_y, 'o-')
    if mark_first_last:
        plt.scatter(ordered_x[0], ordered_y[0], color='g', zorder=100)
        plt.scatter(ordered_x[-2], ordered_y[-2], color='r', zorder=100)

    plt.xlabel('x coordinates')
    plt.ylabel('y coordinates')
    plt.title(f'Cost = {round(cost, 2)}')

    plt.xlim(0, upper_x)
    plt.ylim(0, upper_y)

    plt.show()


def test_cooling_scheme(coord_dict: dict, cooling_scheme: FunctionType,
                        initial_temp: int or float,
                        alphas: list or np.ndarray, n_simulations: int,
                        n_processes: int, mcl=1, convergence_length=1000,
                        max_iterations=60000) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Function for testing simulated annealing of different cooling
    scheme configurations. This function uses the built-in python
    multiprocessing module to perform the tests in parallel.

    Args:
        coord_dict (dict): coordinate dictionary, keys represent the
            city number, while values give the coordinate of that city
        cooling_schedule (FunctionType): function that returns a
            temperature based on an iteration number and a set of
            parameters
        initial_temp (int or float): starting temperature
        alphas (list or np.ndarray): shape parameters to test for the
            cooling scheme function
        n_simulations (int): number of times to repeat a simulation for
            a single alpha value
        n_processes (int): number of CPU cores to use for
            parallelization
        mcl (int, optional): Markov Chain length. Defaults to 1.
        convergence_length (int, optional): number of times a cost
            value has to be repeated for the simulation to stop.
            Defaults to 1000.
        max_iterations (int, optional): maximum number of iterations
            for a simulation to stop. Defaults to 60000.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: array containing the
            costs for all iterations of each simulation, an array
            containing the total iterations for each simulation and an
            array containing all the optimal configurations of each
            simulation
    """
    # Initialize arrays
    costs_array = np.empty((len(alphas), n_simulations, max_iterations*mcl))
    costs_array.fill(np.nan)

    iterations_array = np.empty((len(alphas), n_simulations))
    configs_array = np.empty((len(alphas), n_simulations, len(coord_dict)))

    arguments = []

    # Prepare arguments for the simulated annealing function
    for alpha in alphas:
        for _ in range(n_simulations):
            arguments.append(tuple([coord_dict, cooling_scheme,
                                    initial_temp, alpha, mcl,
                                    convergence_length, max_iterations]))

    # Run the simulated annealing function in parralel for all
    # arguments
    for i in range(0, len(arguments), n_processes):
        if i+n_processes <= len(arguments):
            args = arguments[i:i+n_processes]
        else:
            args = arguments[i:]

        with mp.Pool(n_processes) as pool:
            results = pool.starmap(helper_function, args)

            # Add results to arrays
            for count, result in enumerate(results):
                alpha = args[count][3]
                alpha_index = alphas.index(alpha)

                sim = (i + count) % n_simulations

                iterations_array[alpha_index, sim] = len(result[2])
                configs_array[alpha_index, sim] = result[0]
                costs_array[alpha_index, sim, :len(result[2])] = result[2]

            pool.close()
            pool.join()

        print(f'{i + n_processes} simulations completed')

    return costs_array, iterations_array, configs_array


def helper_function(coord_dict: dict, cooling_scheme: FunctionType,
                    initial_temp: int or float, schedule_parameter: any,
                    mcl: int, convergence_length: int, max_iterations: int
                    ) -> tuple[list, list, list]:
    """Function for passing arguments to simulated annealing solver
    without the use of keyword arguments as this is not compatible
    with parallelization.

    Args:
        coord_dict (dict): coordinate dictionary, keys represent the
            city number, while values give the coordinate of that city
        cooling_schedule (FunctionType): function that returns a
            temperature based on an iteration number and a set of
            parameters
        initial_temp (int or float): starting temperature
        alphas (list or np.ndarray): shape parameters to test for the
            cooling scheme function
        mcl (int): Markov Chain length
        convergence_length (int): number of times a cost value has to
            be repeated for the simulation to stop
        max_iterations (int): maximum number of iterations for a
            simulation to stop

    Returns:
        tuple[list, list, list]: configuration, temperatures, costs
    """
    args = (coord_dict, cooling_scheme, initial_temp, schedule_parameter)
    kwargs = dict(
        show_result=False,
        mcl=mcl,
        convergence_length=convergence_length,
        max_iterations=max_iterations
    )

    config, temperatures, costs \
        = tsp_solver.sim_annealing_solver(*args, **kwargs)

    return config, temperatures, costs


def process_data(data: np.ndarray, confidence: float) -> \
                 tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function for calculating confidence intervals per iteration from
    a given set of simulations. Also provides the average values and
    the last values.

    Args:
        data (np.ndarray): array with axis for the alpha values,
            the different simulations and the different iterations
        confidence (float): confidence level for the confidence
            intervals

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: An array containing
            the average values per alpha value and iteration,
            the confidence interval per alpha value and iteration and
            the last values per alpha ans simulation
    """
    shape = np.shape(data)

    average = np.mean(data, axis=1)
    confidence_intervals = np.empty((shape[0], shape[2], 2))
    confidence_intervals.fill(np.nan)

    last_value_indices = (~np.isnan(data)).sum(axis=2) - 1
    last_values = np.empty(np.shape(last_value_indices))

    for index1 in range(np.shape(last_value_indices)[0]):
        for index2 in range(np.shape(last_value_indices)[1]):
            last_values[index1, index2] = \
                data[index1, index2, last_value_indices[index1, index2]]

    for index1 in range(shape[0]):
        for index2 in range(shape[2]):
            temp_data = data[index1, :, index2]
            intervals = st.norm.interval(confidence=confidence,
                                         loc=np.mean(temp_data),
                                         scale=st.sem(temp_data))

            confidence_intervals[index1, index2, :] = intervals

    return average, confidence_intervals, last_values


def save_data(alphas: list or np.ndarray, avg_costs: np.ndarray,
              conf_interval_costs: np.ndarray, filename: str) -> pd.DataFrame:
    """Saves average costs and confidence interval data in a csv file.

    Args:
        alphas (list or np.ndarray): list of the alpha values. Length
            should be equal to np.shape(avg_costs)[0]
        avg_costs (np.ndarray): 2D numpy array containing the average
            cost values per alpha per iteration
        conf_interval_costs (np.ndarray): 3D numpy array containing the
            confidence intervals (lower, upper) per alpha per iteration
        filename (str): name of the csv file (excluding extension) to
            to which the data will be saved

    Returns:
        pd.DataFrame: DataFrame representation of the data saved in the
            csv file
    """
    labels = ['mu_c', 'CI_c_lower', 'CI_c_upper']
    tuples = itertools.product(labels, map(str, alphas))
    columns = pd.MultiIndex.from_tuples(tuples, names=['variable', 'alpha'])

    data = np.append(avg_costs, conf_interval_costs[:, :, 0], axis=0)
    data = np.append(data, conf_interval_costs[:, :, 1], axis=0)

    df = pd.DataFrame(data.T, columns=columns)
    df.to_csv(f'{filename}.csv')

    return df


def read_data(filename: str) -> pd.DataFrame:
    """Function for converting average cost and confidence interval
    data in a csv file to a pandas DataFrame.

    Args:
        filename (str): filename of the csv file containing the data.
            including extension name.

    Returns:
        pd.DataFrame: DataFrame representation of the data saved in the
            csv file
    """
    df = pd.read_csv(filename, header=[0, 1])
    df = df.drop(columns=('variable', 'alpha'))
    df.columns = df.columns.rename('alpha', level=1)
    df.columns = df.columns.rename('variable', level=0)

    return df


def plot_data(data: pd.DataFrame, labels: list[str], title: str,
              mcl: list[int] = None, figsize: tuple = (6, 4.5),
              zoom_window: bool = False, zoom: int = 2,
              zoom_xlim: tuple = None, zoom_ylim: tuple = None):
    """Plots average cost and confidence interval data contained in a
    pandas DataFrame that was generated by save_data.

    Args:
        data (pd.DataFrame): DataFrame with average cost and confidence
            interval data generated by save_data
        labels (list): legend labels for the graphs
        title (str): plot title
        mcl (list, optional): markov chain lengths for scaling the x
            axis of a perticular graph. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (6, 4.5).
        zoom_window (bool, optional): whether or not to include a
            zoomed in window. Defaults to False.
        zoom (int, optional): zoom factor for the zoom window. Defaults
            to 2.
        zoom_xlim (tuple, optional): x limits of the zoomed in window.
            Defaults to None.
        zoom_ylim (tuple, optional): y limits of the zoomed in window.
            Defaults to None.
    """
    _, ax = plt.subplots(1, 1, figsize=figsize)

    if zoom_window:
        axins = zoomed_inset_axes(ax, zoom, loc='center right')

    _, alphas = zip(*data.columns.to_list())
    alphas = np.unique(alphas)

    if mcl is None:
        mcl = np.ones(len(alphas))

    for index, alpha in enumerate(alphas):
        y = data[('mu_c', alpha)]
        x = np.arange(0, len(y) / mcl[index], 1 / mcl[index])

        y_1 = data[('CI_c_lower', alpha)]
        y_2 = data[('CI_c_upper', alpha)]

        ax.plot(x, y, lw=1, label=labels[index])
        ax.fill_between(x, y_1, y_2, alpha=0.5)

        if zoom_window:
            axins.plot(x, y, lw=1)
            axins.fill_between(x, y_1, y_2, alpha=0.5)

    if zoom_window:
        axins.set(xbound=zoom_xlim, ybound=zoom_ylim)
        axins.tick_params(axis='both', bottom=False, left=False,
                          labelbottom=False, labelleft=False)
        mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.7")

    max_iterations_l = []
    for index, alpha in enumerate(alphas):
        temp_data = data[('mu_c', alpha)]

        max_iterations = len(temp_data.dropna(how='all')) - 1
        max_iterations = max_iterations / mcl[index]
        max_iterations = ceil(max_iterations / 1000) * 1000

        max_iterations_l.append(max_iterations)

    maximum = np.nanmax(data)
    max_iterations = max(max_iterations_l)

    ax.set_ylim(0, ceil(maximum / 100) * 100)
    ax.set_xlim(0, max_iterations)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(title)

    ax.grid(axis='y', alpha=.6)
    ax.legend()

    plt.show()
