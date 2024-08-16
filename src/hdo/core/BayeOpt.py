import torch
import botorch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.piecewise_polynomial_kernel import PiecewisePolynomialKernel
from gpytorch.kernels.rq_kernel import RQKernel
from gpytorch.kernels.polynomial_kernel import PolynomialKernel
from .RandEmbed import random_embedding


def upper_confidence_bound(bounds, model, beta,maximize):
    """
    Basic UCB acquisition function
    :param bounds: the objective function containing the bounds and dimension information
    :param model: the Gaussian Process model, find next point according to this model and UCB
    :param beta: UCB parameter
    :return: next point found by UCB and cost time
    """
    next_point, acq_value_list = optimize_acqf(
        acq_function=UpperConfidenceBound(model=model, beta=beta,maximize=maximize),
        bounds=bounds.t(),
        q=1,
        num_restarts=15,
        raw_samples=512,
    )
    return next_point


def init_points_dataset_bo(n, rand_mat, bounds_low, bounds_high, objective_function):
    """
    random init n points for basic BO
    :param n: init points num
    :param objective_function: the objective function containing the bounds and dimension information
    :return: the original dataset (a dict have 3 elements 'x', 'y' and 'f')
    """
    dim = bounds_low.shape[0]
    dataset = {'y': torch.rand(n, dim) * (bounds_low.t()[1] - bounds_low.t()[0]) + bounds_low.t()[0]}
    # dataset['y'] = dataset['y'].double()
    dataset['x'] = random_embedding(dataset['y'], rand_mat, bounds_high)
    dataset['f'] = objective_function(dataset['x']).reshape(n, 1)
    return dataset


def fit_model_gp(dataset, kernel_type):
    """
    Use training dataset to fit the Gaussian Process Model
    :param dataset: a dict have 2 elements 'x' and 'y', each of them is a tensor shaped (n, dim) and (n, 1)
    :return: the GP model, the marginal log likelihood and cost time
    """
    dataset_x = dataset['y'].clone()
    dataset_f = dataset['f'].clone()

    mean, std = dataset_f.mean(), dataset_f.std()
    # std = 1e-5 if std < 1e-5 else std
    std = 1.0 if std < 1e-6 else std
    dataset_f = (dataset_f - mean) / std

    if kernel_type == 'rbf':
        model = SingleTaskGP(dataset_x, dataset_f, covar_module=ScaleKernel(RBFKernel(ard_num_dims=dataset_x.shape[-1])))
    elif kernel_type == 'matern':
        model = SingleTaskGP(dataset_x, dataset_f)
    elif kernel_type == 'linear':
        model = SingleTaskGP(dataset_x, dataset_f, covar_module=ScaleKernel(LinearKernel()))
    elif kernel_type == 'piece_poly':
        model = SingleTaskGP(dataset_x, dataset_f,
                             covar_module=ScaleKernel(PiecewisePolynomialKernel(ard_num_dims=dataset_x.shape[-1], q=3)))
    elif kernel_type == 'rq':
        model = SingleTaskGP(dataset_x, dataset_f, covar_module=ScaleKernel(RQKernel(ard_num_dims=dataset_x.shape[-1])))
    elif kernel_type == 'poly':
        model = SingleTaskGP(dataset_x, dataset_f,
                             covar_module=ScaleKernel(PolynomialKernel(power=2, ard_num_dims=dataset_x.shape[-1])))
    else:
        print('Please choose the supported kernel method. '
              'We use the default marten kernel to continue the program.')
        model = SingleTaskGP(dataset_x, dataset_f)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_model(mll)
    fit_gpytorch_mll(mll)
    return model, mll


def update_dataset_ucb(new_point_y, new_point_x, value, dataset):
    """
    add (new_point, value) to the normal dataset (NOT COMP DATASET)
    :param new_point_y: shape is (1, dim) or (dim)
    :param value: observed value of new point, shape is (1) or (1, 1)
    :param dataset: the dataset as {(x, f(x))}, which is a dict
    :return: the new dataset
    """
    if not dataset:
        return {'x': new_point_x, 'y': new_point_y, 'f': value}
    if len(value.shape) == 1:
        value = value.reshape(1, -1)
    # if dataset == None:
    #     # if len(value.shape) == 1:
    #     #     value = value.reshape(1, 1)
    #     dataset['x'] = new_point_x
    #     dataset['f'] = value
    #     if new_point_y != None:
    #         if len(new_point_y.shape) == 1:
    #             new_point_y = new_point_y.reshape(1, new_point_y.shape[-1])
    #         dataset['y'] = new_point_y
    #     return dataset
    if new_point_y != None and 'y' in dataset.keys():
        if len(new_point_y.shape) == 1:
            new_point_y = new_point_y.reshape(1, new_point_y.shape[-1])
        # if len(value.shape) == 1:
        #     value = value.reshape(1, 1)
        dataset['y'] = torch.cat([dataset['y'], new_point_y], 0)
        dataset['x'] = torch.cat([dataset['x'], new_point_x], 0)
        # print(1, dataset['f'], dataset['f'].shape)
        # print(2, value, value.shape)
        dataset['f'] = torch.cat([dataset['f'], value], 0)
    else:
        # if len(value.shape) == 1:
        #     value = value.reshape(1, 1)
        dataset['x'] = torch.cat([dataset['x'], new_point_x], 0)

        dataset['f'] = torch.cat([dataset['f'], value], 0)
    return dataset


def next_point_bo(dataset, beta, bounds_low, kernel_type,maximize):
    dataset = dict([(key, dataset[key]) for key in ['y', 'f']])
    try:
        model, _ = fit_model_gp(dataset, kernel_type)
    except RuntimeError:
        print('Cannot fit the GP model, the result is undependable.')
        return False, None
    except botorch.exceptions.errors.ModelFittingError:
        print('Cannot fit the GP model, the result is undependable.')
        return False, None
    try:
        # next_y = upper_confidence_bound(bounds_low, model, beta)
        for i in range(5):
            next_y = upper_confidence_bound(bounds_low, model, beta + 1e-3 * (i + 1),maximize=maximize)
            # print(next_y)
            not_similar_count = 0
            for his_y in dataset['y']:
                temp = next_y - his_y
                # print(1, temp)
                # print(temp.abs().max())
                if temp.abs().max() >= 1e-3:
                    not_similar_count += 1
            # print(not_similar_count, dataset['y'].shape[0])
            if not_similar_count >= dataset['y'].shape[0] * 0.99:
                break
            else:
                print('Already have similar points, re-optimize the acquisition function.')
        # print(next_y)
    except gpytorch.utils.errors.NotPSDError:
        print('The optimization process have converged.')
        return False, None
    except RuntimeError:
        # error = 'RuntimeError'
        print('The optimization process have stopped early, the result is undependable.')
        return False, None
    return True, next_y
