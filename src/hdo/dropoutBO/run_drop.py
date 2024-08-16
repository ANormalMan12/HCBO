import torch
from .dropoutBO import DropoutBO
from ..optimization_problem import OptimizationProblem

def optimize_with_DropoutBO(
    problem:OptimizationProblem,
    num_variables_to_optimize_each_turn:int,
    init_point_budget,
    optimize_budget,
    is_add_ucb
):
    bounds=problem.bounds
    num_vars=problem.bounds.shape[1]
    f=problem.function
    maximize=problem.maximize
    data_x=torch.zeros((init_point_budget,num_vars))
    for j in range(num_vars):
        min_val, max_val = bounds[0][j],bounds[1][j]
        data_x[:, j] = torch.FloatTensor(init_point_budget).uniform_(min_val, max_val)
    data_y=torch.tensor([f(x) for x in data_x]).unsqueeze(1)
    dropout_model=DropoutBO(
        fill_strategy='copy',
        low_d=num_variables_to_optimize_each_turn,
        var_bounds=problem.bounds,
        data_x=data_x,
        data_y=data_y,
        maximize=maximize,
        is_add_ucb=is_add_ucb
    )
    for t in range(optimize_budget):
        try:
            now_x:torch.tensor=dropout_model.predict()
            now_y=f(now_x)
            dropout_model.add_data(now_x,now_y)
        except Exception as e:
            print(e)
            break
    cpu_device=torch.device("cpu")
    X=[x.to(cpu_device) for x in  dropout_model.data_x[init_point_budget:]]
    Y=[float(y.to(cpu_device)) for y in dropout_model.data_y[init_point_budget:]]
    return X,Y