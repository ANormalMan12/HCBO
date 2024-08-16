import torch


def generate_random_matrix(low_dim, high_dim, sigma):
    """
    generate random matrix shape as (low_dim, high_dim)
    :param low_dim: dimension of low dim space
    :param high_dim: dimension of high dim space
    :param sigma: exp times
    :return: random matrix
    """
    rand_mat = torch.randn(low_dim, high_dim)
    # rand_mat = torch.randn(low_dim, high_dim, dtype=torch.double)
    for i in range(high_dim):
        rand_mat[:, i] = rand_mat[:, i] * sigma[i]
    return rand_mat

def random_embedding(y, rand_mat, bounds_high):
    """
    embedding low dimension vector y into original solution space
    :param y: low dimension vector y
    :param rand_mat: random matrix
    :param bounds_high: bounds for objection function
    :return: high dimension vector x
    """
    x = torch.mm(y, rand_mat)
    #mid_val=bounds_high.t().sum(dim=0) / 2
    #x = x + mid_val
    # print(bounds.t().sum(dim=0) / 2)
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         if x[i, j] < bounds_high[j, 0]:
    #             x[i, j] = bounds_high[j, 0]
    #         elif x[i, j] > bounds_high[j, 1]:
    #             x[i, j] = bounds_high[j, 1]
    for dim in range(x.shape[1]):
        x[:, dim] = torch.clamp(x[:, dim], bounds_high[dim, 0], bounds_high[dim, 1])
    return x


def random_projection(x, rand_mat_inv, bounds_high):
    x = x - bounds_high.t().sum(dim=0) / 2
    y = torch.mm(x, rand_mat_inv)
    return y