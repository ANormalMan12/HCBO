###################################################
# This file is to test the effective dimension of MSLR dataset
###################################################

import torch
import numpy as np
from cmaes import CMA
import random
# from Fit import Net
from .Gtopx import gtopx
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
bounds = np.array([[-1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05,
                    1.15, 1.7, -np.pi, -np.pi, -np.pi, -np.pi],
                   [0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 6.0, 6.0, 6.5,
                    291.0, np.pi, np.pi, np.pi, np.pi]])


# find the maximum and minimum of the function
def find_max_min(dim_chosen, x_mean):
    dim = len(dim_chosen)
    print(dim)
    print("finding min")
    min_optimizer = CMA(mean=x_mean[dim_chosen], sigma=1.0, bounds=bounds.T[dim_chosen, :], population_size=20)
    global_min = 1e10
    for generation in range(500):
        solutions = []
        for _ in range(min_optimizer.population_size):
            x = min_optimizer.ask()
            x_model = x_mean.copy()
            for idx, dim_i in enumerate(dim_chosen):
                x_model[dim_i] = x[idx]
            # x_model = torch.from_numpy(x_model)
            x_model = x_model.tolist()
            # x_model = torch.as_tensor(x_model, dtype=torch.float32).to(device)
            # value = model(x_model).cpu().detach().numpy()
            # print(gtopx(2, x_model, 1, 22, 0))
            # print(np.array(x_model) <= bounds[1])
            # print(np.array(x_model) >= bounds[0])
            [value, function_g] = gtopx(2, x_model, 1, 22, 0)
            value = np.array(value)
            # value = 4.0 * value
            solutions.append((x, value))
            if value <= global_min:
                global_min = value
                x_opt = x
        min_optimizer.tell(solutions)
    print("finding max")
    max_optimizer = CMA(mean=x_mean[dim_chosen], sigma=1.3, bounds=bounds.T[dim_chosen, :])
    global_max = 1e10
    for generation in range(500):
        solutions = []
        for _ in range(max_optimizer.population_size):
            x = max_optimizer.ask()
            x_model = x_mean.copy()
            for idx, dim_i in enumerate(dim_chosen):
                x_model[dim_i] = x[idx]
            # x_model = torch.from_numpy(x_model)
            x_model = x_model.tolist()
            # x_model = torch.as_tensor(x_model, dtype=torch.float32).to(device)
            # value = -model(x_model).cpu().detach().numpy()
            # print(gtopx(2, x_model, 1, 22, 0))
            [value, function_g] = gtopx(2, x_model, 1, 22, 0)
            value = -np.array(value)
            # value = 4.0 * value
            solutions.append((x, value))
            if value <= global_max:
                global_max = value
                x_opt = x
        max_optimizer.tell(solutions)
    global_max = -global_max
    return global_min, global_max


# device = torch.device("cuda:0")
# model_path = 'model.pt'
# model = torch.load(model_path)

# device = torch.device('cpu')
# model_path = 'model.pt'
# model = torch.load(model_path, map_location=torch.device('cpu'))

# cassini2 mean datas

# x_mean = np.array([-500., 4., 0.5, 0.5, 250., 300., 265., 1000., 1500., 0.45, 0.45, 0.45, 0.45, 0.45,
#                    3.5, 3.5, 3.8, 146.35, 0., 0., 0., 0.])
x_mean = bounds.mean(axis=0)
# print(x_mean)
file = open('cassini_eff_dim_min_opt_20.txt', 'w')
global_max, global_min = -1e10, 1e10
for i in range(50):
    min_val, max_val = find_max_min([i for i in range(22)], x_mean.copy())
    global_min = min(min_val, global_min)
    global_max = max(max_val, global_max)
print(f'global max = {global_max}, global min = {global_min}')

# test the affect of each dimension
# points = np.ones(10000)
# points = torch.ones(10000)
# torch_mean = torch.from_numpy(x_mean)
# torch_mean = torch.as_tensor(torch_mean, dtype=torch.float32)
points = x_mean.reshape(1, 22)
# print(points)
# print(np.repeat(points, 3, axis=0))
points = np.repeat(points, 10000, axis=0)
print(points[0])
points = points.reshape(10000, 22)
diff = np.zeros(22)
min_value = np.zeros(22)
mean_eff = []
min_eff = []

file.write(f'=====================the effect of every dimension=============================\n')
for i in range(22):
    points_now = points.copy()
    change = np.linspace(bounds[0, i], bounds[1, i], 10000)
    points_now[:, i] = change
    # result = model(points_now.to(device)).cpu()
    result = np.zeros([points_now.shape[0], 1])
    for j in range(points_now.shape[0]):
        # print(gtopx(2, points_now[j], 1, 22, 0))
        [f, g] = gtopx(2, points_now[j], 1, 22, 0)
        result[j] = f[0]
    # resul = 4.0 * result
    dim_max = np.max(result)
    dim_min = np.min(result)
    diff[i] = dim_max - dim_min
    min_value[i] = dim_min
print(f'the function value change in different dimension :\n{diff}')
file.write(f'the function value change in different dimension :{diff.tolist()}\n')
print(f'the function minimum in different dimension :\n{min_value}')
file.write(f'the function minimum in different dimension :{min_value.tolist()}\n')
print(f'global max = {global_max}, global min = {global_min}')
file.write(f'global max = {global_max}, global min = {global_min}\n')
diff = diff / (float(global_max) - float(global_min))
print(f"the percentage of function value change in different dimension :\n{diff}")
file.write(f"the percentage of function value change in different dimension :{diff.tolist()}\n")
print(f"the function minimum in different dimension :\n{min_value}")
file.write(f"the function minimum in different dimension :{min_value.tolist()}\n")
mean_eff.append(diff.mean())
min_eff.append(min_value.mean())

# test the affect of different subspaces, repeat iter_nums times
dim_num = [i for i in range(2, 22, 1)]
iter_nums = 20
for chosen_dim_num in dim_num:
    file.write(f'=====================number of chosen dimensions {chosen_dim_num}=============================\n')
    diff = np.zeros(iter_nums)
    min_value = np.zeros(iter_nums)
    for i in range(iter_nums):
        chosen_dim = random.sample([i for i in range(22)], chosen_dim_num)
        chosen_dim.sort()
        dim_min, dim_max = find_max_min(chosen_dim, x_mean.copy())
        diff[i] = dim_max - dim_min
        min_value[i] = dim_min
        file.write(f'in iteration {i}, the chosen dimensions are :{chosen_dim}\n')

    print(f'the function value change in different dimension :{diff}')
    file.write(f'the function value change in different dimension :\n{diff.tolist()}\n')
    print(f'global max = {global_max}, global min = {global_min}')
    file.write(f'global max = {global_max}, global min = {global_min}\n')
    diff = diff / (float(global_max) - float(global_min))
    print(f"the percentage of function value change in different dimension :{diff}")
    file.write(f"the percentage of function value change in different dimension :\n{diff.tolist()}\n")
    print(f"the mean of the percentage of function value change in different dimension :{diff.mean()}")
    file.write(f"the mean of the percentage of function value change in different dimension :{diff.mean()}\n")
    print(f"the function minimum in different dimension :\n{min_value}")
    file.write(f"the function minimum in different dimension :{min_value.tolist()}\n")
    mean_eff.append(diff.mean())
    min_eff.append(min_value.mean())
min_eff.append(global_min.__float__())
file.write('=====================summary=============================\n')
file.write(f"the mean of the percentage of function value change in different dimension :\n{mean_eff}\n")
file.write(f"the mean of the function minimum in different dimension :\n{min_eff}\n")
