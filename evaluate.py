import os.path
from time import process_time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hopefield import solve_hopefield
from ipsolver import solve_ip
from generate_cases import read_test_case, EXHAUSTIVE_SEARCH_LIMIT, TestCase, plot_case

plt.rcParams['figure.figsize'] = (10, 20)


def get_tour_length(test_case: TestCase, hopefield_activations: np.array) -> float:
    length = 0.
    for i in range(test_case.cities):
        prv = np.argmax(hopefield_activations[:, i])
        nxt = np.argmax(hopefield_activations[:, (i+1) % test_case.cities])  # prevent overflow
        length += test_case.distances[prv][nxt]
    return length


def plot_evaluation(
        name: str,
        test_case: TestCase,
        hopefield_activations: np.array,
        hopefield_optimum: float,
        ip_x: np.array,
        ip_optimum: float
):
    fig, axes = plt.subplots(2, 1)

    # Train data + Hopefield
    axes[0].scatter(test_case.points[:, 0], test_case.points[:, 1])
    # if test_case.cities < EXHAUSTIVE_SEARCH_LIMIT:
    #     # Plot best route calculated by exhaustive search for small enough samples
    #     axes[0].plot(
    #         [*test_case.points[test_case.best_route, 0], test_case.points[test_case.best_route[0], 0]],
    #         [*test_case.points[test_case.best_route, 1], test_case.points[test_case.best_route[0], 1]],
    #         c='r'
    #     )
    if test_case.cities < EXHAUSTIVE_SEARCH_LIMIT:
        hopefield_relative_error = (hopefield_optimum / test_case.optimum - 1) * 100
        axes[0].set_title(f'Hopefield prediction (error={hopefield_relative_error:.2f} %)')
    else:
        axes[0].set_title(f'Hopefield prediction (Hopefield optimum={hopefield_optimum:.2f})')
    xs = test_case.points[np.argmax(hopefield_activations, axis=0), 0]
    ys = test_case.points[np.argmax(hopefield_activations, axis=0), 1]
    axes[0].plot(
        [*xs, xs[0]],
        [*ys, ys[0]],
        c='g',
        linestyle='dashed',
        linewidth=2
    )
    # Train data + IP solver
    axes[1].scatter(test_case.points[:, 0], test_case.points[:, 1])
    # if test_case.cities < EXHAUSTIVE_SEARCH_LIMIT:
    #     axes[1].plot(
    #         [*test_case.points[test_case.best_route, 0], test_case.points[test_case.best_route[0], 0]],
    #         [*test_case.points[test_case.best_route, 1], test_case.points[test_case.best_route[0], 1]],
    #         c='r'
    #     )
    if test_case.cities < EXHAUSTIVE_SEARCH_LIMIT:
        ip_relative_error = (ip_optimum / test_case.optimum - 1) * 100
        axes[1].set_title(f'IP solver prediction (error={ip_relative_error:.2f} %)')
    else:
        axes[1].set_title(f'IP solver prediction (OPT={ip_optimum:.2f} %)')

    src = 0
    dst = -1
    while dst != 0:
        dst = np.argmax(ip_x[src, :])
        axes[1].plot(test_case.points[[src, dst], 0], test_case.points[[src, dst], 1], c='b', linestyle='dashed')
        src = dst
    fig.savefig(f'{name}.png')


def main():
    test_names = []
    hopefield_times = []
    ip_times = []
    hopefield_errors = []
    ip_errors = []
    hopefield_optimums = []
    ip_optimums = []

    df = pd.DataFrame(columns=[
        'test_name', 'hopefield_time', 'ip_time', 'hopefield_error', 'ip_error', 'hopefield_optimum', 'ip_optimum'
    ])
    df.to_csv('results.csv')

    for s in [5, 6, 7, 8, 9, 10, 15, 20, 40]:
        for c in [0, 1]:
            test_case = read_test_case(os.path.join('test_cases', f'{s}C{c}.txt'))
            print(f'TEST CASE {s}C{c}\n---------------------------------------')
            t_start_hopefield = process_time()  # fractional seconds
            hopefield_activations = solve_hopefield(test_case)
            t_end_hopefield = process_time()  # fractional seconds
            print(f'Hopefield time:\t\t{t_end_hopefield - t_start_hopefield:.4f} s')
            hopefield_times.append(t_end_hopefield - t_start_hopefield)
            hopefield_optimum = get_tour_length(test_case, hopefield_activations)
            if s < EXHAUSTIVE_SEARCH_LIMIT:
                hopefield_relative_error = (hopefield_optimum / test_case.optimum - 1) * 100
                print(f'Hopefield error:\t{hopefield_relative_error:.2f} %')
                hopefield_errors.append(hopefield_relative_error)
            print(f'Hopefield optimum:\t{hopefield_optimum:.4f}')
            hopefield_optimums.append(hopefield_optimum)
            t_start_ip = process_time()  # fractional seconds
            ip_x, ip_optimum = solve_ip(test_case)
            t_end_ip = process_time()  # fractional seconds
            print(f'IP solver time:\t\t{t_end_ip - t_start_ip:.4f} s')
            ip_times.append(t_end_ip - t_start_ip)
            if s < EXHAUSTIVE_SEARCH_LIMIT:
                ip_relative_error = (ip_optimum / test_case.optimum - 1) * 100
                print(f'IP solver error:\t{ip_relative_error:.2f} %')
                ip_errors.append(ip_relative_error)
            print(f'IP solver optimum:\t{ip_optimum:.4f}')
            ip_optimums.append(ip_optimum)
            print('---------------------------------------')
            plot_evaluation(f'{s}C{c}', test_case, hopefield_activations, hopefield_optimum, ip_x, ip_optimum)
            test_names.append(f'{s}C{c}')

            df = pd.DataFrame(columns=[
                'test_name', 'hopefield_time', 'ip_time', 'hopefield_error', 'ip_error', 'hopefield_optimum',
                'ip_optimum'
            ],
                data=[[test_names[-1], hopefield_times[-1], ip_times[-1], hopefield_errors[-1], ip_errors[-1],
                      hopefield_optimums[-1], ip_optimums[-1]]]
            )
            df.to_csv('results.csv', header=False, mode='a')


if __name__ == '__main__':
    main()


