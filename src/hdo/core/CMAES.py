from cmaes import CMA
# import cma
import numpy as np
import torch
from . import ObjFunc
import os
from .BayeOpt import init_points_dataset_bo, random_embedding
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class cmaes():
    def __init__(self, d, pop_size, bounds, sigma=1.3, seed=0):
        self.model = CMA(mean=np.zeros(d), sigma=sigma, bounds=bounds, seed=seed, population_size=pop_size)
        self.best_x = None
        self.best_f = 1e5

    def gen_next_point(self):
        points = [self.model.ask() for _ in range(self.model.population_size)]
        return True, points, [torch.unsqueeze(torch.from_numpy(point), dim=0).float() for point in points]

    def update(self, points, values):
        solutions = [(points[i], values[i]) for i in range(self.model.population_size)]
        for i, value in enumerate(values):
            if value < self.best_f:
                self.best_f = value
                self.best_x = points[i]
        self.model.tell(solutions)
# class cmaes():
#     def __init__(self, pop_size, bounds, budget, init_data, sigma=1., seed=0):
#         np.random.seed(seed)
#         cma_opts = {
#             'seed': seed,
#             'popsize': pop_size,
#             'maxiter': budget,
#             'verbose': -1,
#             'bounds': [-1 * float(bounds[0][-1]), 1 * float(bounds[0][-1])]
#         }
#         n = torch.argmax(init_data['f'])
#         self.model = cma.CMAEvolutionStrategy(init_data['y'][n].tolist(), sigma, inopts=cma_opts)
#         self.model.ask()
#         self.model.tell(init_data['y'].numpy(), (-init_data['f']).reshape(1, -1).numpy()[0])
#         # self.model = CMA(mean=np.zeros(d), sigma=sigma, bounds=bounds, seed=seed, population_size=pop_size)
#         # self.best_x = None
#         # self.best_f = 1e5
#
#     def gen_next_point(self):
#         points = self.model.ask()
#         # points = [self.model.ask() for _ in range(self.model.population_size)]
#         return True, points, [torch.unsqueeze(torch.from_numpy(point), dim=0).float() for point in points]
#
#     def update(self, points, values):
#         self.model.tell(points, values)

        # solutions = [(points[i], values[i]) for i in range(self.model.population_size)]
        # for i, value in enumerate(values):
        #     if value < self.best_f:
        #         self.best_f = value
        #         self.best_x = points[i]
        # self.model.tell(solutions)

from ..optimization_problem import OptimizationProblem
def optimize_with_CMAES(
    problem:OptimizationProblem,
    sigma,
    optimize_budget
):
    fucntion=problem.function
    
    pop_size = 20
    bounds=np.array(problem.bounds.t())
    dim_high=bounds.shape[1]
    optimizer = CMA(mean=bounds.mean(axis=1), sigma=sigma, bounds=bounds, population_size=pop_size)
    x_data = []
    y_data = []
    for generation in range(optimize_budget // pop_size + 1):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = float(fucntion(torch.tensor(x)))
            solutions.append((x, value))
        optimizer.tell(solutions)
        x_data += [solution[0] for solution in solutions]
        y_data += [solution[1] for solution in solutions]

    return x_data,y_data