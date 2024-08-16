from ..utils import *

def find_EI_acq_val(model:SingleTaskGP,bounds,maximize, best_f,**kwargs):
    return optimize_acqf(
        acq_function=ExpectedImprovement(model, best_f,maximize=maximize),
        bounds=bounds,
        q=1,
        num_restarts=ACQ_NUM_RESTARTS,
        raw_samples=ACQ_RAW_SAMPLES,
    )
def find_GP_UCB_acq_val(model:SingleTaskGP,bounds,maximize,**kwargs):
    return optimize_acqf(
        acq_function=UpperConfidenceBound(model,UCB_BETA,maximize=maximize),
        bounds=bounds,
        q=1,
        num_restarts=ACQ_NUM_RESTARTS,
        raw_samples=ACQ_RAW_SAMPLES,
    )
def find_PI_acq_val(model:SingleTaskGP,bounds,maximize, best_f,**kwargs):
    return optimize_acqf(
        acq_function=ProbabilityOfImprovement(model, best_f,maximize=maximize),
        bounds=bounds,
        q=1,
        num_restarts=ACQ_NUM_RESTARTS,
        raw_samples=ACQ_RAW_SAMPLES,
    )

def get_x_acq_item(acqusition_function,BOmodelItem,maximize,global_best_y=None):
    if(global_best_y is None):
        best_y=BOmodelItem.data_y.max().item() if maximize else BOmodelItem.data_y.min().item()
    else:
        best_y=global_best_y
    return acqusition_function(
        BOmodelItem.model,
        BOmodelItem.var_bounds,
        maximize=maximize,
        best_f=best_y
    )
