import torch
import numpy as np
import zoopt
from zoopt.utils.zoo_global import nan
from .Gtopx import gtopx
# from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset as CEMBLD
# from design_bench.oracles.sklearn import RandomForestOracle as RFO


# domain [-32.768, 32.768]
def ackley_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    # x.shape = [nums, dim]
    # x = 32.768 * x
    d_e = 1
    # K = 100 * x.shape[1]
    K = 10000
    # x = x - 32.768 / 2  #，
    x = x - 10
    #new_a=extract_eff_dim(x,d_e,random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    # dim = x[:, 0:d_e].shape[1]
    dim = d_e
    a = eff_dim ** 2
    #print("a==",a)
    a = a.sum(dim=1) / dim
    a = -0.2 * torch.sqrt(a)
    #b = torch.cos(2 * torch.pi * x[:, 0:d_e])
    b = torch.cos(2 * torch.pi * eff_dim)
    b = b.sum(dim=1) / dim
    dis = x[:, :] ** 2

    dis = (dis.sum(dim=1) - (eff_dim ** 2).sum(dim=1)) / K
    # print(a.shape)
    return -(- 20 * torch.exp(a) - torch.exp(b) + 20 + torch.exp(torch.tensor(1)) + dis)


# domain [-32.768, 32.768]
def ackley(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    # x.shape = [nums, dim]
    # x = 32.768 * x
    d_e = 10
    # K = 100 * x.shape[1]
    K = 10000
    x = x - 32.768 / 2
    # x = x[:, 0:d_e]
    dim = x[:, 0:d_e].shape[1]
    # print(x)
    a = x[:, 0:d_e] ** 2
    a = a.sum(dim=1) / dim
    a = -0.2 * torch.sqrt(a)
    b = torch.cos(2 * torch.pi * x[:, 0:d_e])
    b = b.sum(dim=1) / dim
    dis = x[:, d_e:] ** 2
    dis = dis.sum(dim=1) / K
    # print(a.shape)
    return -(- 20 * torch.exp(a) - torch.exp(b) + 20 + torch.exp(torch.tensor(1)) + dis)


# domain [-1, 1]
def ackley_reshape(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    # x.shape = [nums, dim]
    x = 32.768 * x
    d_e = 10
    x = x - 32.768 / 2
    x = x[:, 0:d_e]
    dim = x.shape[1]
    # print(x)
    a = x ** 2
    a = a.sum(dim=1) / dim
    a = -0.2 * torch.sqrt(a)
    b = torch.cos(2 * torch.pi * x)
    b = b.sum(dim=1) / dim
    # print(a.shape)
    return -(- 20 * torch.exp(a) - torch.exp(b) + 20 + torch.exp(torch.tensor(1)))


# domain [0, 1]
def hartmann6(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 6
    x = x[:, 0:d_e]
    n = x.shape[0]
    alpha = torch.tensor([1., 1.2, 3., 3.2])
    A = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])
    result = torch.zeros(n)
    for i in range(4):
        temp = torch.zeros(n)
        for j in range(6):
            # print(x[:, j] - P[i, j])
            # print((x[:, j] - P[i, j]) ** 2)
            # print(A[i, j] * (x[:, j] - P[i, j]) ** 2)
            temp += A[i, j] * (x[:, j] - P[i, j]) ** 2
        result += alpha[i] * torch.exp(-temp)
    return result


# domain [-1, 1]
def hartmann6_reshape(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = (x + 1) / 2
    d_e = 6
    x = x[:, 0:d_e]
    n = x.shape[0]
    alpha = torch.tensor([1., 1.2, 3., 3.2])
    A = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])
    result = torch.zeros(n)
    for i in range(4):
        temp = torch.zeros(n)
        for j in range(6):
            temp += A[i, j] * (x[:, j] - P[i, j]) ** 2
        result += alpha[i] * torch.exp(-temp)
    return result


# domain [[-5, 10], [0, 15]]
def branin(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    x1 = x[:, 0]
    x2 = x[:, 1]
    a = 1
    b = 5.1 / (4 * torch.pi ** 2)
    c = 5 / torch.pi
    r = 6
    s = 10
    t = 1 / (8 * torch.pi)
    return -(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s)


# domain [-1, 1]
def branin_reshape(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    x1 = (x[:, 0] + 1) * 7.5 - 5
    x2 = (x[:, 1] + 1) * 7.5
    a = 1
    b = 5.1 / (4 * torch.pi ** 2)
    c = 5 / torch.pi
    r = 6
    s = 10
    t = 1 / (8 * torch.pi)
    return -(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s)


# domain [-10, 10]
def levy(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    w = 1 + (x - 1) / 4
    w = w[:, 0:d_e]
    a = torch.sin(torch.pi * w[:, 0]) ** 2
    c = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)
    b = (w[:, 1:-1] - 1) ** 2 * (1 + torch.sin(torch.pi * w[:, 1:-1] + 1) ** 2)
    b = b.sum(dim=1)
    dis = (x[:, d_e:] - 1) ** 2
    dis = dis.sum(dim=1) / K
    return - (a + b + c + dis)


# domain [-1, 1]
def levy_reshape(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = x.clone() * 10
    d_e = 10
    K = 10000
    w = 1 + (x - 1) / 4
    w = w[:, 0:d_e]
    a = torch.sin(torch.pi * w[:, 0]) ** 2
    c = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)
    b = (w[:, 1:-1] - 1) ** 2 * (1 + torch.sin(torch.pi * w[:, 1:-1] + 1) ** 2)
    b = b.sum(dim=1)
    dis = (x[:, d_e:] - 1) ** 2
    dis = dis.sum(dim=1) / K
    return - (a + b + c + dis)


# domain [-10, 10]
def dixon_price(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    a = (x[:, 0] - 1) ** 2
    b = torch.tensor([i for i in range(2, d_e+1)]) * (2 * x[:, 1:d_e] ** 2 - x[:, 0:d_e-1]) ** 2
    b = b.sum(dim=1)
    dis = (x[:, d_e:] - 1) ** 2
    dis = dis.sum(dim=1) / K
    return -(a + b + dis) / 1e4


# domain [-50, 50]
def griewank(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = x - 10
    d_e = 10
    K = 10000
    a = x[:, 0:d_e] ** 2 / 4000
    a = a.sum(dim=1)
    b = torch.ones(x.shape[0])
    for i in range(d_e):
        b *= torch.cos(x[:, i] / torch.sqrt(torch.tensor(i+1)))
    dis = (x[:, d_e:] - 10) ** 2
    dis = dis.sum(dim=1) / K
    return -(a - b + 1 + dis)


# domain [-50, 50]
def griewank_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = x - 10
    d_e = 10
    K = 10000
    #new_a=extract_eff_dim(x,d_e,random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2 / 4000
    a = a.sum(dim=1)
    b = torch.ones(x.shape[0])
    for i in range(d_e):
        b *= torch.cos(eff_dim[:, i] / torch.sqrt(torch.tensor(i + 1)))
    dis = x[:, :] ** 2
    dis1 = eff_dim[:, :] ** 2    #dis
    dis = (dis.sum(dim=1) - dis1.sum(dim=1)) / K
    return -(a - b + 1 + dis)



# domain [-5.12, 5.12]
def sphere(x):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    x = x - 1
    a = x[:, 0:d_e] ** 2
    a = a.sum(dim=1)
    dis = x[:, d_e:] ** 2
    dis = dis.sum(dim=1) / K
    return -(a + dis)


# domain [-5.12, 5.12]
def sphere_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    x = x - 1
    #new_a = extract_eff_dim(x,d_e,random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K
    return -(a + dis)


# domain [-5, 10]
def zakharov_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    x = x - 5

    # new_a = extract_eff_dim(x,d_e,random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    par = torch.zeros(x.shape[0])
    for i in range(d_e):
        par += 0.5 * (i + 1) * eff_dim[:, i]
    b = par ** 2
    c = par ** 4
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K

    return -(a + b + c + dis)


# domain [-5, -10]
def rosenbrock_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    b = eff_dim[:, 1:] - eff_dim[:, :d_e-1] ** 2
    b = 100 * (b ** 2)
    b = b + (eff_dim[:, 1:] - 1) ** 2
    b = b.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K
    return -(b + dis)


def cassini2_gtopx(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan

    res = torch.zeros(x.shape[0], 1)
    for i in range(x.shape[0]):
        [f, g] = gtopx(2, x[i].tolist(), 1, x.shape[1], 0)
        res[i] = f[0]

    return -res


# dataset = CEMBLD()
# oracle = RFO(dataset, noise_std=0.0)
# bound = torch.tensor([[12, 161, 184, 184, 45, 119, 60, 60, 184, 119, 184, 184, 60, 184, 184, 401, 115, 84, 84, 84, 85,
#                        85, 85, 84, 60, 60, 60, 60, 48, 48, 13],
#                       [12, 15, 15, 15, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
#
# def chembl(x, random_list):
#     x = x.clone()
#     if len(x.shape) == 1:
#         x = x.unsqueeze(dim=0)
#     x = np.array(x.numpy(), dtype=np.int32)
#     y = oracle.predict(x)
#     y = torch.from_numpy(y)
#     y = torch.as_tensor(y, dtype=torch.float32)
#     n = y.shape[0]
#     y = y.resize(1, n)
#     return y[0]

