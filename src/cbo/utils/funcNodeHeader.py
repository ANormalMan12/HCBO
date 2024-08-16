import numpy as np
import random
import os
import pandas as pd
class funcNode():
    def __init__(self,Ntype:str,kRange:list,bRange:list,NarguSet:set):
        """_summary_
            ，，
            

        Args:
            Ntype (str): :
                'L':Y=kX+b
                'P':Y=k*X^2+b
                'S'sin:Y=k*sin(X)+b
            NarguDict (set): ，
        """
        self.funcType=Ntype
        self.K=np.random.uniform(kRange[0], kRange[1], (len(NarguSet)))
        self.b=random.uniform(bRange[0],bRange[1])
        self.arguSet=NarguSet # previous nodes, function argument
    def __call__(self,eps,**sampleSet):
        if(len(self.arguSet)==0):
            return self.b
        X=np.hstack(sampleSet[c] for c in self.arguSet)#*np.ones((1,1))*
        # sampleSet
        if(self.funcType=="L"):
            return np.float64(self.K.dot(X)+self.b)
        elif(self.funcType=="P"):
            return np.float64(self.K.dot((X*X))+self.b)
        elif(self.funcType=='S'):
            return np.float64(self.K.dot((np.sin(X)))+self.b)
        else:
            raise BaseException("Problem on Func node")
    def print(self):
        print(self.funcType,end=';')#
        print('k:',self.k,end=',')
        print('b:',self.b,end=',')
        print("Prev:",end='')#
        for prev in self.arguSet:
            print((repr(prev)),end=' ')
        print()

def get_obs_trueobs(experiment):
    try:
        if os.path.exists('./Data/' + str(experiment) + '/' + 'observations_latest.csv'):
            full_observational_samples = pd.DataFrame(pd.read_csv('./Data/' + str(experiment) + '/' + 'observations_latest.csv', low_memory=False))
        else:
            full_observational_samples = pd.read_pickle('./Data/' + str(experiment) + '/' + 'observations.pkl')
    except:
        full_observational_samples=None
    try:
        true_observational_samples = pd.read_pickle('./Data/' + str(experiment) + '/' + 'true_observations.pkl')
    except:
        true_observational_samples=None
    return full_observational_samples,true_observational_samples
