"""Module for solving Travelling Salesman Problems using simulated
annealing."""

# Authors: Nathanyel Schut and Frenk Klein Schiphorst
# Date: 19-12-2022

from math import ceil, exp, sqrt
from types import FunctionType

import numpy as np
import pandas as pd


def eucl_dist(x_1: int or float, y_1: int or float,
              x_2: int or float, y_2: int or float) -> float:
    """Calculates euclidean distance between two points.

    Args:
        x1 (int or float): x-coordinate of point 1
        y_1 (int or float): y-coordinate of point 1
        x_2 (int or float): x-coordinate of point 2
        y_2 (int or float): y-coordinate of point 2

    Returns:
        float: distance between the given points
    """
    return sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)


def cost_func(config: list or np.ndarray, coord_dict: dict) -> float:
    """Calculates the cost or total distance for a given configuration.

    Args:
        config (ArrayLike): configuration, given by a list or array of
            cities in order of their visit
        coord_dict (dict): coordinate dictionary, keys represent the
            city number, while values give the coordinate of that city

    Returns:
        float: total cost of the configuration
    """
    cost = 0

    # Add distances to the cost variable
    for i in range(len(config)-1):
        x_1, y_1 = coord_dict[config[i]]
        x_2, y_2 = coord_dict[config[i+1]]

        cost += eucl_dist(x_1, y_1, x_2, y_2)

    # Complete the circle
    x_1, y_1 = coord_dict[config[-1]]
    x_2, y_2 = coord_dict[config[0]]

    cost += eucl_dist(x_1, y_1, x_2, y_2)

    return cost


def permutate(config: list or np.ndarray) -> list:
    """Permutate a given configuration by removing two non-adjacent
    links and reconnecting a different way.

    Args:
        config (ArrayLike): configuration, given by a list or array of
            cities in order of their visit

    Returns:
        list: new, reordered configuration
    """
    config_copy = list(config).copy()
    n_cities = len(config_copy)

    # Find non-adjacent random links to remove, denoted by the first
    # city of the link Non-adjacent then means that the second link
    # number cannot come before or after the first link number in the
    # configuration.

    link1_index = np.random.randint(low=0, high=n_cities-1)
    link2_index = np.random.randint(low=0, high=n_cities-1)

    while (link2_index == link1_index + 1 or link2_index == link1_index - 1 or
           link2_index == link1_index):
        link2_index = link2_index = np.random.randint(low=0, high=n_cities-1)

    # Ensure link 2 comes later in the configuration
    if link1_index > link2_index:
        link1_index, link2_index = link2_index, link1_index

    # The cities from the second city of link one to the first city of
    # link two should be reversed in order due to the new links that
    # are formed. This comes from the fact that the new links can only
    # be remade in 1 configuration as we would otherwise get 2 closed
    # loops.

    for i in range(ceil(((link2_index - link1_index) / 2) - 1)+1):
        tmp = config_copy[link1_index+1+i]
        config_copy[link1_index+1+i] = config_copy[link2_index-i]
        config_copy[link2_index-i] = tmp

    return config_copy


def sim_annealing_solver(coord_dict: dict, cooling_schedule: FunctionType,
                         initial_temp: int or float, schedule_parameter: any,
                         show_result=True, mcl=1, convergence_length=1000,
                         max_iterations=60000) -> np.ndarray or tuple:
    """Travelling Salesman Problem solver using simulated annealing.
    Stopping criteria are: costs remain the same for n iterations where
    n is the convergence length, max_iterations is reached or the
    temperature drops to 0 or below.

    Args:
        coord_dict (dict): coordinate dictionary, keys represent the
            city number, while values give the coordinate of that city
        cooling_schedule (FunctionType): function that returns a
            temperature based on an iteration number and a set of
            parameters
        initial_temp (intorfloat): starting temperature
        schedule_parameter (any): shape parameter for the cooling
            scheme function
        show_results (bool, default=True): whether to print a summary
            of the results of the simulated annealing process. Defaults
            to True.
        mcl (int, optional): Markov Chain length. Defaults to 1.
        convergence_length (int, optional): number of times a cost
            value has to be repeated for the simulation to stop.
            Defaults to 1000.
        max_iterations (int, optional): maximum number of iterations
            for a simulation to stop. Defaults to 60000.

    Returns:
        np.ndarray: optimal configuration
        np.ndarray, np.ndarray (if save_all_steps=True): optimal
            configuration, numpy array containing the iteration,
            temperature and cost at each step of the process. stored in
            an array with shape (iterations, 3)
    """
    costs = []
    temperatures = []

    # Initial configuration
    n_cities = len(coord_dict)
    config = np.arange(1, n_cities+1, 1)
    np.random.shuffle(config)
    cost = cost_func(config, coord_dict)

    convergence = False
    conv_test_indices = np.arange(-mcl*convergence_length, -mcl, mcl)

    i = 0
    temp_i = cooling_schedule(i, initial_temp, schedule_parameter)

    # Annealing steps
    while not convergence and temp_i > 0 and i < max_iterations:

        # Repeat step until the markov chain length is reached
        for _ in range(mcl):
            new_config = permutate(config)
            new_cost = cost_func(new_config, coord_dict)

            # Accept if new cost is lower
            if new_cost < cost:
                config = new_config
                cost = new_cost

            # Else accept with acceptance probability alpha
            else:
                try:
                    alpha = exp((cost-new_cost) / temp_i)
                except ZeroDivisionError:
                    alpha = exp((cost-new_cost) / (10**(-10)))

                u = np.random.uniform()

                if u < alpha:
                    config = new_config
                    cost = new_cost

            costs.append(cost)
            temperatures.append(temp_i)

        # Check for convergence
        if (i >= convergence_length - 1 and
                all(np.array(costs)[conv_test_indices] == costs[-mcl])):
            convergence = True

        i += 1

        temp_i = cooling_schedule(i, initial_temp, schedule_parameter)

    # Print overview of the results
    if show_result:
        print('Results:')

        columns = ['Iteration', 'Temperature', 'Cost']

        iterations = list(range(len(temperatures)))
        results = np.array([iterations, temperatures, costs]).T
        results_df = pd.DataFrame(results, columns=columns)

        indices = np.linspace(0, len(temperatures) - 1, 10, dtype=int)
        results_df = results_df.iloc[indices]

        results_df = results_df.set_index('Iteration')

        print(results_df)

    return config, temperatures, costs
