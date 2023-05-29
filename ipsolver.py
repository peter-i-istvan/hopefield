import os
from itertools import product

import numpy as np
from mip import Model, xsum, minimize, BINARY

from generate_cases import TestCase, read_test_case


def solve_ip(test_case: TestCase):
    model = Model()
    city_indices = range(test_case.cities)
    x = [[model.add_var(var_type=BINARY) for _ in city_indices] for _ in city_indices]
    y = [model.add_var() for _ in city_indices]
    model.objective = minimize(xsum(test_case.distances[i][j] * x[i][j] for i in city_indices for j in city_indices))
    for i in city_indices:
        model += xsum(x[i][j] for j in city_indices if j != i) == 1
        model += xsum(x[j][i] for j in city_indices if j != i) == 1
    for (i, j) in product(city_indices[1:], city_indices[1:]):
        if i != j:
            model += y[i] - (test_case.cities + 1) * x[i][j] >= y[j] - test_case.cities

    model.verbose = 0
    model.optimize()

    if model.num_solutions:
        x = [[var.x for var in row] for row in x]
        return np.array(x), model.objective_value
    else:
        return None, None


def main():
    test_case = read_test_case(os.path.join('test_cases', '20C0.txt'))
    v, obj = solve_ip(test_case)
    print(v)
    print(obj)


if __name__ == '__main__':
    main()
