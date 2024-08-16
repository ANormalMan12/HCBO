from ...sem import *
from ...bo_component import *
from botorch.models.transforms.input import Normalize
def optimize_cgo_with_HCBO(
    target_sem:SEM,
    set_selector:SetSelectorAbstract,
    val_acq_function,
    BOmodel_class,
    es:exploration_set,
    init_intervention_data,
    optimization_cost:int
):
    hcbo_optimizer=HCBO_Optimizer(#Should Use the same kind of Init Data
        set_selector=set_selector,
        val_acqusition_function=val_acq_function,
        es=es,
        init_intervention_data=init_intervention_data,
        BO_model_class=BOmodel_class,
        the_sem=target_sem
    )
    
    result_x_list,result_y_list,visit_IS_list,ISSF_info=hcbo_optimizer(target_sem,optimization_cost)
    assert(len(result_x_list)==len(result_y_list))
    assert(len(visit_IS_list)==len(result_y_list))
    return result_x_list,result_y_list,visit_IS_list,ISSF_info

import time

class HCBO_Optimizer():
    def __init__(
        self,
        set_selector,
        val_acqusition_function,
        es,
        BO_model_class,
        init_intervention_data,
        the_sem,
        use_global_best_y_for_acq_set=True
        ):
        self.use_global_best_y_for_acq_set=use_global_best_y_for_acq_set
        self.set_selector:SetSelector_BO=set_selector
        self.val_acqusition_function=val_acqusition_function
        self.es=es
        self.BO_model_class=BO_model_class
        self.BO_model_list=[]
        self.maximize=(the_sem.task=='max')
        if(self.maximize):
            self.global_best_y=-float("inf")
        else:
            self.global_best_y=float("inf")
        for i,interv_set in enumerate(self.es):
            print("init BO for",i,interv_set)
            data_x_y=init_intervention_data[interv_set]
            data_x=torch.tensor(data_x_y[0])
            data_y=torch.tensor(data_x_y[1])
            bounds=the_sem.get_bounds(interv_set)
            self.BO_model_list.append(
                self.BO_model_class(
                    variables=interv_set,
                    var_bounds=bounds,
                    data_x=data_x,
                    data_y=data_y
                )
            )
            if(self.maximize):
                self.global_best_y=max(self.global_best_y,float(data_y.max()))
            else:
                self.global_best_y=min(self.global_best_y,float(data_y.min()))
    def __call__(
        self,
        target_sem:SEM,
        optimize_cost:int
    ):
            
        maximize=(target_sem.task=='max')
        last_index=-1
        step=0
        total_cost=0
        result_x_list=[]
        result_y_list=[]
        visit_IS_list=[]
        for i,bo_model in enumerate(self.BO_model_list):
            IS_size=bo_model.var_bounds.shape[1]
            now_x_list=bo_model.data_x.tolist()
            num_samples=len(now_x_list)
            result_x_list+=now_x_list
            result_y_list+=bo_model.data_y.squeeze(1).tolist()
            visit_IS_list+=[self.es[i] for k in range(num_samples)]
            total_cost+=num_samples*IS_size
        print("Initial Total Cost",total_cost)
        start_time = time.time()
        while(total_cost<=optimize_cost):
            current_time = time.time()
            if current_time - start_time >=MAX_TIME_LIMIT:#！！！！！
                break#！！！！！！！
            print(step,":",total_cost)
            try:  
                interv_index=self.set_selector.predict_set(
                    BOmodelList=self.BO_model_list,
                    maximize=maximize,
                    last_index=last_index,
                    global_best_y=self.global_best_y if self.use_global_best_y_for_acq_set else None)
                #nUCB doesn't need global best, only EI needs it
                
                interv_set=self.es[interv_index]
                try:
                    #if(self.BO_model_class==BOmodelNormalized):
                    #    value_acq_GP=SingleTaskGP(
                    #        covar_module=gpytorch.kernels.RBFKernel(),
                            #covar_module=gpytorch.kernels.AdditiveStructureKernel(
                            #    base_kernel=gpytorch.kernels.RBFKernel(),
                            #    num_dims=len(self.es[interv_index])
                            #),
                    #        train_X=self.BO_model_list[interv_index].data_x.to(GLOBAL_DEVICE),
                    #        train_Y=self.BO_model_list[interv_index].data_y.to(GLOBAL_DEVICE),
                            #input_transform=Normalize(
                            #    len(self.es[interv_index]),
                            #    bounds=self.BO_model_list[interv_index].var_bounds
                            #)
                    #    )
                    #    mll = ExactMarginalLogLikelihood(value_acq_GP.likelihood, value_acq_GP)
                    #    fit_gpytorch_mll(mll)
                    #else:
                    value_acq_GP=self.BO_model_list[interv_index].model
                    new_x,new_acq_val=self.val_acqusition_function(# Optimize Local Model
                        model=value_acq_GP,
                        bounds=self.BO_model_list[interv_index].var_bounds,
                        maximize=maximize,
                        best_f=max(self.BO_model_list[interv_index].data_y) if maximize
                                else min(self.BO_model_list[interv_index].data_y),
                    )
                except Exception as e:
                    print(e)
                    print("May Fail to fit GP on",interv_set,".Disable it and re-run the turn")
                    self.set_selector.disable_index(interv_index)
                    while(last_index==interv_index):
                        last_index=random.randint(0,len(self.es)-1)
                    continue
                last_index=interv_index
                interv_plan=[]
                for i,var in enumerate(interv_set):
                    interv_plan.append((var,new_x[0][i]))
                new_y=target_sem.intervene(
                    1,SEM_REPEATED_TIMES,interv_plan
                )
                #print(self.es[interv_index])
                #print(target_sem.get_bounds(self.es[interv_index]))
                #print(new_x)
                #print(new_y)
                if(self.maximize):
                    self.global_best_y=max(self.global_best_y,new_y[0])
                else:
                    self.global_best_y=min(self.global_best_y,new_y[0])
                
                self.BO_model_list[interv_index].add_data(
                    torch.tensor(new_x),
                    torch.tensor([new_y])
                )
            #print(step,new_x,new_y)
            except Exception as e:
                print(e)
                break
            visit_IS_list.append(self.es[interv_index])
            result_y_list.append(new_y[0])
            result_x_list.append(new_x[0])
            total_cost+=len(new_x[0])
            step+=1
        return result_x_list,result_y_list,visit_IS_list,self.set_selector.get_info()