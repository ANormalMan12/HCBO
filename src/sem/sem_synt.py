from .sem import *
#from .synthetic_datasets import *
#from .sem_nodes import *



def check_bounds(theSEM:SEM,interv:interv_plan):
    interv_s=[]
    x=[]
    for var,val in interv:
        interv_s.append(var)
        x.append(val)
    past_interv_s=tuple(interv_s)
    interv_s=tuple(sorted(interv_s))
    assert(past_interv_s==interv_s)
    bounds=theSEM.get_bounds(interv_s)
    for k in range(len(interv_s)):
        if(x[k]<bounds[0][k] or x[k]>bounds[1][k]):
            return False
    return True

"""
class SEM_synt(SEM):
    def __init__(
            self,
            name:str,
            W,
            connectionT,
            intervenable_variable_list:list,
            function_type:str="linear",
            noise_type:str='lingam',
            task:str='max',
            normalize_strategy="NO") -> None:
        self.name=name
        self._W=W# W's meaning is based on _connection's transpose
        self._connectionT=connectionT
        self._connection=connectionT.T
        self._intervenable_variable_list=intervenable_variable_list
        self._function_type=function_type
        self._normalize_type_per_node=get_normalize_funcs(dim=self._connection.shape[0],
                                                          strategy=normalize_strategy)
        self._noise_type=noise_type
        self.task=task
        self.intervention_buffer={}
    def __str__(self):
        return self.name+'-'+self._function_type+'-'+self._noise_type+'-'+self.task
    def get_intervenable_variables(self):
        return self._intervenable_variable_list
    #return n_samplesY
    def sample(self,n_samples:int):
        return gen_data_given_model(
            self._W,n_samples=n_samples,normalize_list=self._normalize_type_per_node,
            noise_type=self._noise_type,only_return_y_mean=False
        )
    def intervene(self,n_samples:int,repeated_times:int,interv:interv_plan):
        assert(check_bounds(self,interv))
        if(self._function_type=="linear"):
            gen_data_func=gen_data_given_model
        #elif(self.inter_type=='2nd'):
        #    gen_data_func=gen_data_given_model_2nd_order
        else:
            raise ValueError("_function_type not supported")
        ParaPool=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)
        delay_gen_data=delayed(gen_data_func)
        Ys=ParaPool(
            delay_gen_data(
                b=self._W,
                n_samples=repeated_times,
                normalize_list=self._normalize_type_per_node,
                noise_type=self._noise_type,
                interv=interv)
            for i in range(n_samples))
        return Ys

    def get_data(self, intervention_set:interv_set):
        return copy.deepcopy(self.intervention_buffer[intervention_set])

    def get_all_data(self):
        return copy.deepcopy(self.intervention_buffer)
    def get_connection(self):
        return self._connection

    def fill_synt_data(self,es_list:exploration_set,
            allow_replace:bool=False,
            repeated_times:int=SEM_REPEATED_TIMES,
            ):
        for intervention_set in es_list:
            if(not allow_replace and intervention_set in self.intervention_buffer):
                raise ValueError("Try to add synthetic data when allow_replace is False")
            data_x=[]
            data_y=[]
            for j in range(SYNT_INIT_INTERVENTION_DATA_NUM):
                intervention_plan=get_random_intervention_plan(
                    intervention_set,
                    min_vals=self.get_bounds(intervention_set)[0],
                    max_vals=self.get_bounds(intervention_set)[1]
                )
                x_value=[]
                for var,val in intervention_plan:
                    x_value.append(val)
                y_value=self.intervene(
                    n_samples=1,
                    repeated_times=repeated_times,
                    interv=intervention_plan
                )
                data_x.append(x_value)
                data_y.append(y_value)
            self.intervention_buffer[intervention_set]=[data_x,data_y]

"""
def get_pars_list(_connection):
    pars_list=[]
    for i in range(_connection.shape[0]):
        pars_list.append([])
    for i in range(_connection.shape[0]):
        for j in range(_connection.shape[1]):
            if(_connection[i,j]):
                pars_list[j].append(i)
    return pars_list
"""
class SEM_synt_new(SEM):
    def __init__(self,
            name:str,
            connectionT,
            intervenable_variable_list:list,
            pars_list,
            function_node_list,
            noise_type:str='gaussian',
            task:str='max',
    ):
        self.name=name
        self._connection=connectionT.T
        self.task=task
        self._intervenable_variable_list=intervenable_variable_list
        self.noise_type=noise_type
        self.pars_list=pars_list
        
        self._funciton_node_list=function_node_list
        
        
    def __str__(self):
        return self.name
    def get_intervenable_variables(self):
        return self._intervenable_variable_list

    def sample(self,n_samples:int):
        return self._gene_data(n_samples,None,False)

    def _gene_data(self,n_samples:int,interv:interv_plan,only_return_y_mean):
        n_vars=self._connection.shape[0]
        X = np.zeros((n_samples,n_vars))
        s = np.ones([n_vars])
        if(self.noise_type=='gaussian'):
            ss = np.random.randn(n_samples, n_vars) * s
        else:
            raise ValueError("noise_type not supported")
        j=0
        for i in range(n_vars):
            if(interv is not None and j<len(interv) and i==interv[j][0]):
                X[:,i]=interv[j][1]
                j+=1
                continue
            result=self._funciton_node_list[i](X[:,self.pars_list[i]])
            X[:,i]=result+ss[:,i]
        if(interv is not None):#!ADD
            assert(j==len(interv))
        if(only_return_y_mean):
            return X[:, -1].mean()
        return X
    def intervene(self,n_samples:int,repeated_times:int,interv:interv_plan):
        assert(check_bounds(self,interv))
        ParaPool=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)
        delay_gen_data=delayed(self._gene_data)
        Ys=ParaPool(
            delay_gen_data(
                n_samples=repeated_times,
                interv=interv,
                only_return_y_mean=True)
            for i in range(n_samples))
        return Ys
    def get_data(self, intervention_set:interv_set):
        return copy.deepcopy(self.intervention_buffer[intervention_set])
    def get_all_data(self):
        return copy.deepcopy(self.intervention_buffer)
    def get_connection(self):
        return self._connection

    def fill_synt_data(self,es_list:exploration_set,
        allow_replace:bool=False,
        repeated_times:int=SEM_REPEATED_TIMES,
        ):
        if(allow_replace):
            self.intervention_buffer={}
        for intervention_set in es_list:
            if(not allow_replace and intervention_set in self.intervention_buffer):
                raise ValueError("Try to add synthetic data when allow_replace is False")
            data_x=[]
            data_y=[]
            for j in range(SYNT_INIT_INTERVENTION_DATA_NUM):
                intervention_plan=get_random_intervention_plan(
                    intervention_set,
                    min_vals=self.get_bounds(intervention_set)[0],
                    max_vals=self.get_bounds(intervention_set)[1]
                )
                x_value=[]
                for var,val in intervention_plan:
                    x_value.append(val)
                y_value=self.intervene(
                    n_samples=1,
                    repeated_times=repeated_times,
                    interv=intervention_plan
                )
                data_x.append(x_value)
                data_y.append(y_value)
            self.intervention_buffer[intervention_set]=[data_x,data_y]
"""
class SEM_synt(SEM):
    def __init__(self,
        name:str,
        connectionT,
        intervenable_variable_list:list,
        pars_list,
        function_node_list,
        noise_type:str='gaussian',
        task:str='max',
    ):
        self.name=name
        self._connection=connectionT.T
        self.task=task
        self._intervenable_variable_list=intervenable_variable_list
        self.noise_type=noise_type
        self.pars_list=pars_list
        self._funciton_node_list=function_node_list
        
    def __str__(self):
        return self.name
    def get_intervenable_variables(self):
        return sorted(list(self._intervenable_variable_list))

    def sample(self,n_samples:int):
        return self._gene_data(n_samples,None,False)

    def _gene_data(self,n_samples:int,interv:interv_plan,only_return_y_mean,with_observe_noise=True):
        n_vars=self._connection.shape[0]
        X = np.zeros((n_samples,n_vars))
        j=0
        for i in range(n_vars):
            if(interv is not None and j<len(interv) and i==interv[j][0]):
                X[:,i]=interv[j][1]
                j+=1
                continue
            result=self._funciton_node_list[i](X[:,self.pars_list[i]]).reshape(n_samples)
            U=np.random.normal(size=n_samples)# Assume an additive exogenous variable distribution
            X[:,i]=result+U
        if(interv is not None):
            assert(j==len(interv))
        

        if(with_observe_noise):
            s = np.ones([n_samples,n_vars])
            if(self.noise_type=='gaussian'):
                ss = np.random.randn(n_samples,n_vars) * s
            else:
                raise ValueError("noise_type not supported")
            X=X+ss
        if(only_return_y_mean):
            return X[:, -1].mean()
        return X
    
    def intervene(self,n_samples:int,repeated_times:int,interv:interv_plan):
        assert(check_bounds(self,interv))
        ParaPool=Parallel(n_jobs=GLOBAL_PARALLEL_N_JOBS)
        delay_gen_data=delayed(self._gene_data)
        Ys=ParaPool(
            delay_gen_data(
                n_samples=repeated_times,
                interv=interv,
                only_return_y_mean=True)
            for i in range(n_samples))
        return Ys
    #def get_data(self, intervention_set:interv_set):
    #    return copy.deepcopy(self.intervention_buffer[intervention_set])
    #def get_all_data(self):
    #    return copy.deepcopy(self.intervention_buffer)
    def get_connection(self):
        return self._connection

def get_synt_intervention_data(
        the_sem,es_list:exploration_set,
        init_synt_data_num_per_subset,
        
    ):
    repeated_times:int=SEM_REPEATED_TIMES
    intervention_buffer={}
    for intervention_set in es_list:
        print("Begin",intervention_set)
        data_x=[]
        data_y=[]
        for j in range(init_synt_data_num_per_subset):
            intervention_plan=get_random_intervention_plan(
                intervention_set,
                min_vals=the_sem.get_bounds(intervention_set)[0],
                max_vals=the_sem.get_bounds(intervention_set)[1]
            )
            x_value=[]
            for var,val in intervention_plan:
                x_value.append(val)
            y_value=the_sem.intervene(
                n_samples=1,
                repeated_times=repeated_times,
                interv=intervention_plan
            )
            data_x.append(x_value)
            data_y.append(y_value)
        intervention_buffer[intervention_set]=[data_x,data_y]
    return intervention_buffer

from scipy import stats
def sigmoid(X):
    return 1.0/(1.0+np.exp(-X))
class HealthSEM(SEM_synt):
    def __init__(self):
        self._connection=np.zeros((9,9),dtype=bool)
        self.num_name=['A','B','C',"H","W","I","S","P","Y"]
        self.AgeIndex=0
        self.BM_RIndex=1
        self.CIIndex=2
        self.HeiIndex=3
        self.WeiIndex=4
        self.BMIindex=5
        self.StaIndex=6
        self.AspIndex=7
        self.YIndex=8
        
        self._connection[self.BM_RIndex,self.WeiIndex]=True
        self._connection[self.CIIndex,self.WeiIndex]=True
        
        self._connection[self.HeiIndex,self.WeiIndex]=True
        self._connection[self.HeiIndex,self.BMIindex]=True
        
        self._connection[self.WeiIndex,self.BMIindex]=True

        self._connection[self.AgeIndex,self.StaIndex]=True
        self._connection[self.AgeIndex,self.WeiIndex]=True
        self._connection[self.AgeIndex,self.AspIndex]=True
        self._connection[self.AgeIndex,self.YIndex]=True
        
        self._connection[self.BMIindex,self.StaIndex]=True
        self._connection[self.BMIindex,self.AspIndex]=True
        self._connection[self.BMIindex,self.YIndex]=True

        self._connection[self.AspIndex,self.YIndex]=True
        self._connection[self.StaIndex,self.YIndex]=True
        #'A' is age
        #'B' is BMR
        #'C' is CI
        #'
        self.name="HealthGraph"
        self.task='min'
        self._intervenable_variable_list=[
            self.CIIndex,self.StaIndex,self.AspIndex
        ]
        self.MIS=get_power_set_without_empty(self._intervenable_variable_list)
        num_feature=self._connection.shape[0]
        init_bounds=[[0.0 for i in range(num_feature)],[0.0 for i in range(num_feature)]]
        init_bounds=torch.tensor(init_bounds)
        self.noise_type='gaussian'
    def __str__(self):
        return self.name
    def get_intervenable_variables(self):
        return sorted(list(self._intervenable_variable_list))

    def iterate_health_once(self,X,i):
        ret_size=X.shape[0]
        if(i==self.AgeIndex):
            return np.random.uniform(low=55, high=75,size=ret_size)  # age
        elif(i==self.CIIndex):
            return np.random.uniform(low=-100, high=100,size=ret_size)
        elif(i==self.BM_RIndex):
            return stats.truncnorm.rvs(-1, 2,size=ret_size) * 10 + 1500
        elif(i==self.HeiIndex):
            return stats.truncnorm.rvs(-0.5, 0.5,size=ret_size) * 10 + 175
        elif(i==self.WeiIndex):
            return (
                X[:,self.BM_RIndex] + 6.8 * X[:,self.AgeIndex] - 5 * X[:,self.HeiIndex]) / (13.7 + X[:,self.CIIndex] * 150. / 7716) # weight
        elif(i==self.BMIindex):
            return X[:,self.WeiIndex] / ((X[:,self.HeiIndex] / 100)**2)  # bmi
        elif(i==self.AspIndex):
            return sigmoid(-8.0 + 0.10 * X[:,self.AgeIndex] + 0.03  * X[:,self.BMIindex])  # aspirin
        elif(i==self.StaIndex):
            return sigmoid(-13.0 + 0.10 * X[:,self.AgeIndex] + 0.20  * X[:,self.BMIindex])
        elif(i==self.YIndex):
            return  (6.8 + 0.04 * X[:,self.AgeIndex] - 0.15 * X[:,self.BMIindex] - 0.60 * X[:,self.StaIndex] +
            0.55 * X[:,self.AspIndex] + sigmoid(2.2 - 0.05 *X[:,self.AgeIndex] + 0.01 * X[:,self.BMIindex]
                    - 0.04 * X[:,self.StaIndex] + 0.02 * X[:,self.AspIndex])
                    +np.random.normal(0,0.4,size=ret_size) 
            )
        else:
            raise ValueError("Not legal input index")
    def _gene_data(self,n_samples:int,interv:interv_plan,only_return_y_mean,with_observe_noise=True):
        n_vars=self._connection.shape[0]
        X = np.zeros((n_samples,n_vars))
        j=0
        for i in range(n_vars):
            if(interv is not None and j<len(interv) and i==interv[j][0]):
                X[:,i]=interv[j][1]
                j+=1
                continue
            X[:,i]=self.iterate_health_once(X,i)
        if(interv is not None):
            assert(j==len(interv))
        

        if(with_observe_noise):
            s = np.ones([n_samples,n_vars])
            if(self.noise_type=='gaussian'):
                ss = np.random.randn(n_samples,n_vars) * s
            else:
                raise ValueError("noise_type not supported")
            X=X+ss
        if(only_return_y_mean):
            return X[:, -1].mean()
        return X

