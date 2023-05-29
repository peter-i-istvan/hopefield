import numpy as np
from itertools import permutations
from typing import Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os


@dataclass
class TestCase:
    cities: int
    points: np.array
    distances: np.array
    best_route: Optional[np.array]
    optimum: Optional[float]

    def save_to_file(self, file):
        file.write(str(self.cities) + '\n')
        file.write(f'\n'.join(f'{x:.6f} {y:.6f}' for x, y in self.points) + '\n')
        file.write(f'\n'.join([' '.join(f'{d:.6f}' for d in row) for row in self.distances]) + '\n')
        if self.best_route is not None:
            file.write(' '.join(f'{p}' for p in self.best_route) + '\n')
        else:
            file.write('-1' + '\n')
        if self.optimum is not None:
            file.write(f'{self.optimum:.6f}' + '\n')
        else:
            file.write('-1' + '\n')


def read_test_case(in_file_path) -> TestCase:
    with open(in_file_path) as f:
        cities = int(f.readline())

        points = []
        for _ in range(cities):
            coordinates = f.readline().split(' ')
            x, y = float(coordinates[0]), float(coordinates[1])
            points.append((x, y))
        points = np.array(points)

        distances = []
        for _ in range(cities):
            line = f.readline()
            dists = line.split(' ')
            dists = [float(dist) for dist in dists]
            distances.append(dists)
        distances = np.array(distances)

        best_route = f.readline().split(' ')
        if int(best_route[0]) < 0:  # No optimum provided
            best_route = None
            optimum = None
        else:
            best_route = [int(p) for p in best_route]
            optimum = float(f.readline())

        return TestCase(cities, points, distances, best_route, optimum)


EXHAUSTIVE_SEARCH_LIMIT = 11


def generate_case(size: int) -> TestCase:
    points = np.random.random((size, 2))
    distances = np.array([[np.linalg.norm(p - q) for p in points] for q in points])
    # distances[range(size), range(size)] = 100.0
    best_route = None
    optimum = None
    if size < EXHAUSTIVE_SEARCH_LIMIT:
        optimum = size * np.sqrt(2)  # highest total distance on the unit square
        for perm in permutations(range(size)):
            length = np.sum(distances[perm[:-1], perm[1:]])
            length += distances[perm[-1], perm[0]]
            if length < optimum:
                optimum = length
                best_route = perm
    return TestCase(size, points, distances, best_route, optimum)


def plot_case(test_case: TestCase) -> None:
    plt.scatter(test_case.points[:, 0], test_case.points[:, 1])
    if test_case.optimum is not None:
        plt.plot(
            [*test_case.points[test_case.best_route, 0], test_case.points[test_case.best_route[0], 0]],
            [*test_case.points[test_case.best_route, 1], test_case.points[test_case.best_route[0], 1]],
            c='r'
        )
    plt.title(f'N={test_case.cities} cities')


def main():
    if not os.path.isdir('test_cases'):
        os.mkdir('test_cases')

    for size in [5, 6, 7, 8, 9, 10, 15, 20, 40]:
        for case in [0, 1]:
            test_case = generate_case(size)  # Generate
            with open(os.path.join('test_cases', f'{size}C{case}.txt'), 'w') as f:
                test_case.save_to_file(f)  # Save data
            plot_case(test_case)
            plt.savefig(os.path.join('test_cases', f'{size}C{case}.png'))  # Save plot
            plt.clf()


if __name__ == '__main__':
    main()
