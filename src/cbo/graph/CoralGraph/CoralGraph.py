from .CoralGraph_DoFunctions import *
from .CoralGraph_CostFunctions import define_costs
from ..AbstractGraph import *
class CoralGraph():
    """
    An instance of the class graph giving the graph structure in the Coral reef example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples, true_observational_samples):
        self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]
        self.N = np.asarray(observational_samples['N'])[:,np.newaxis]
        self.X = np.asarray(observational_samples['X'])[:,np.newaxis]
        self.T = np.asarray(observational_samples['T'])[:,np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:,np.newaxis]
        self.P = np.asarray(observational_samples['P'])[:,np.newaxis]
        self.O = np.asarray(observational_samples['O'])[:,np.newaxis]
        self.S = np.asarray(observational_samples['S'])[:,np.newaxis]
        self.L = np.asarray(observational_samples['L'])[:,np.newaxis]
        self.E = np.asarray(observational_samples['E'])[:,np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:,np.newaxis]

        true_Y = np.asarray(true_observational_samples['Y'])[:,np.newaxis]
        true_N = np.asarray(true_observational_samples['N'])[:,np.newaxis]
        true_X = np.asarray(true_observational_samples['X'])[:,np.newaxis]
        true_T = np.asarray(true_observational_samples['T'])[:,np.newaxis]
        true_D = np.asarray(true_observational_samples['D'])[:,np.newaxis]
        true_P = np.asarray(true_observational_samples['P'])[:,np.newaxis]
        true_O = np.asarray(true_observational_samples['O'])[:,np.newaxis]
        true_S = np.asarray(true_observational_samples['S'])[:,np.newaxis]
        true_L = np.asarray(true_observational_samples['L'])[:,np.newaxis]
        true_E = np.asarray(true_observational_samples['E'])[:,np.newaxis]
        true_C = np.asarray(true_observational_samples['C'])[:,np.newaxis]

        ## Fit SEM with true_observation，CoralDataset
        self.reg_Y = LinearRegression().fit(np.hstack((true_L, true_N, true_P, true_O, true_C, true_X, true_E)), true_Y)
        self.reg_P = LinearRegression().fit(np.hstack((true_S,true_T, true_D, true_E)), true_P)
        self.reg_O = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_E)), true_O)
        self.reg_X = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_E)), true_X)
        self.reg_T = LinearRegression().fit(true_S, true_T)
        self.reg_D = LinearRegression().fit(true_S, true_D)
        self.reg_C = LinearRegression().fit(np.hstack((true_N, true_L, true_E)), true_C)
        self.reg_S = LinearRegression().fit(true_E, true_S)
        self.reg_E = LinearRegression().fit(true_L, true_E)

        ## Define distributions for the exogenous variables
        params_list = scipy.stats.gamma.fit(true_L)  # gamma
        self.dist_Light = scipy.stats.gamma(a = params_list[0], loc = params_list[1], scale = params_list[2])

        mixture = sklearn.mixture.GaussianMixture(n_components=3) # 
        # 
        mixture.fit(true_N)
        # N
        self.dist_Nutrients_PC1 = mixture
        # 
        self.define_connection()

    def define_connection(self):
        self.full_var=['N', 'L', 'E', 'C', 'S', 'T', 'D', 'P', 'O', 'X']
        self.num_name=['N', 'L', 'E', 'C', 'S', 'T', 'D', 'P', 'O', 'X','Y']
        self.name_num = {}
        for i in range(len(self.num_name)):
            self.name_num[self.num_name[i]] = i
        self.is_manipulaitve = np.zeros((len(self.name_num),))  #! 
        self.is_manipulaitve[self.name_num['N']] = 1
        self.is_manipulaitve[self.name_num['O']] = 1
        self.is_manipulaitve[self.name_num['C']] = 1
        self.is_manipulaitve[self.name_num['T']] = 1
        self.is_manipulaitve[self.name_num['D']] = 1
        self.is_manipulaitve[self.name_num['Y']] = 1  #! Y，
        self.connection = np.zeros((len(self.name_num), len(self.name_num)))
        self.connection[self.name_num['L']][self.name_num['Y']] = 1
        self.connection[self.name_num['L']][self.name_num['E']] = 1
        self.connection[self.name_num['L']][self.name_num['C']] = 1
        self.connection[self.name_num['N']][self.name_num['Y']] = 1
        self.connection[self.name_num['N']][self.name_num['C']] = 1
        self.connection[self.name_num['P']][self.name_num['Y']] = 1
        self.connection[self.name_num['O']][self.name_num['Y']] = 1
        self.connection[self.name_num['C']][self.name_num['Y']] = 1
        self.connection[self.name_num['S']][self.name_num['O']] = 1
        self.connection[self.name_num['S']][self.name_num['P']] = 1
        self.connection[self.name_num['S']][self.name_num['X']] = 1
        self.connection[self.name_num['S']][self.name_num['D']] = 1
        self.connection[self.name_num['S']][self.name_num['T']] = 1
        self.connection[self.name_num['T']][self.name_num['O']] = 1
        self.connection[self.name_num['T']][self.name_num['P']] = 1
        self.connection[self.name_num['T']][self.name_num['X']] = 1
        self.connection[self.name_num['D']][self.name_num['P']] = 1
        self.connection[self.name_num['D']][self.name_num['O']] = 1
        self.connection[self.name_num['D']][self.name_num['X']] = 1
        self.connection[self.name_num['X']][self.name_num['Y']] = 1
        self.connection[self.name_num['E']][self.name_num['Y']] = 1
        self.connection[self.name_num['E']][self.name_num['P']] = 1
        self.connection[self.name_num['E']][self.name_num['O']] = 1
        self.connection[self.name_num['E']][self.name_num['C']] = 1
        self.connection[self.name_num['E']][self.name_num['S']] = 1
        self.connection[self.name_num['E']][self.name_num['X']] = 1
    """
    def define_SEM(self):
        "Simulate SEM with Causal Structures"
        def fN(epsilon, **kwargs):
            return self.dist_Nutrients_PC1.sample(1)[0][0][0]

        def fL(epsilon, **kwargs):
            return self.dist_Light.rvs(1)[0]

        def fE(epsilon, L, **kwargs):
            X = np.ones((1,1))*L
            return np.float64(self.reg_E.predict(X))

        def fC(epsilon, N, L, E, **kwargs):
            X = np.ones((1,1))*np.hstack((N, L, E))
            return np.float64(self.reg_C.predict(X))
            #return value

        def fS(epsilon, E, **kwargs):
            X = np.ones((1,1))*E
            return np.float64(self.reg_S.predict(X))
            #return value

        def fT(epsilon, S, **kwargs):
            X = np.ones((1,1))*S
            return np.float64(self.reg_T.predict(X))
            #return value

        def fD(epsilon, S, **kwargs):
            X = np.ones((1,1))*S
            return np.float64(self.reg_D.predict(X))
            #return value

        def fP(epsilon, S, T, D, E, **kwargs):
            X = np.ones((1,1))*np.hstack((S,T, D, E))
            return np.float64(self.reg_P.predict(X))
            #return value

        def fO(epsilon, S, T, D, E, **kwargs):
            X = np.ones((1,1))*np.hstack((S,T, D, E))
            return np.float64(self.reg_O.predict(X))
            #return value

        def fX(epsilon, S, T, D, E, **kwargs):
            X = np.ones((1,1))*np.hstack((S, T, D, E))
            return np.float64(self.reg_X.predict(X))
            #return value

        def fY(epsilon, L, N, P, O, C, X, E, **kwargs):
            X = np.ones((1,1))*np.hstack((L, N, P, O, C, X, E))
            return np.float64(self.reg_Y.predict(X)) 
            #return value

        graph = OrderedDict ([
          ('N', fN),
          ('L', fL),
          ('E', fE),
          ('C', fC),
          ('S', fS),
          ('T', fT),
          ('D', fD),
          ('P', fP),
          ('O', fO),
          ('X', fX),
          ('Y', fY)
        ])

        return graph
    """
    def __str__():
        return "CoralGraph"

    def get_sets(self):

        MIS_1 = [['N'], ['C'], ['T'], ['D'],['O']]
        MIS_2 = [['N', 'O'], ['N', 'C'], ['N', 'T'], ['N', 'D'], ['O', 'C'], ['O', 'T'], ['O', 'D'], ['T', 'C'], ['T', 'D'], ['C', 'D']]
        MIS_3 = [['N', 'O', 'C'], ['N', 'O', 'T'], ['N', 'O', 'D'], ['N', 'C', 'T'], ['N', 'C', 'D'], ['N','T', 'D'], 
         ['O', 'C', 'T'], ['O', 'C', 'D'], ['C', 'T', 'D'], ['O','T', 'D']]
        MIS_4 = [['N','O','C','T'], ['N','O','C','D'], ['N','O','T','D'], ['N','T','D','C'], ['T','D','C','O']]
        MIS_5 = [['N','O','C','T','D']]

        MIS = MIS_1 + MIS_2 + MIS_3
        BO = MIS_1 + MIS_2 + MIS_3 + MIS_4 + MIS_5


        POMIS = MIS

        manipulative_variables = ['N', 'O', 'C', 'T', 'D']
        return {"MIS":MIS,"POMIS": POMIS,"BO":BO,"manu_var":manipulative_variables}


    def get_set_BO(self):
        manipulative_variables = ['N', 'O', 'C', 'T', 'D']
        return manipulative_variables

    def get_interventional_ranges(self):
        min_intervention_N = 0
        max_intervention_N = 10

        min_intervention_O = 6
        max_intervention_O = 15

        min_intervention_D = 1750
        max_intervention_D = 2400

        min_intervention_T = 1800
        max_intervention_T = 3000

        min_intervention_C = -10
        max_intervention_C = 0

#-------------------------------------
        min_intervention_X = 210
        max_intervention_X = 600

        min_intervention_L = 350
        max_intervention_L = 1400

        min_intervention_P = 7
        max_intervention_P = 9

        min_intervention_S = 33
        max_intervention_S = 40

        min_intervention_E = 15
        max_intervention_E = 35


        # min_intervention_N = -2 
        # max_intervention_N = 5

        # min_intervention_O  = 2
        # max_intervention_O = 4

        # min_intervention_C = 0
        # max_intervention_C = 1

        # min_intervention_T = 2400
        # max_intervention_T = 2500

        # min_intervention_D = 1950
        # max_intervention_D = 2100

        dict_ranges = OrderedDict ([
          ('N', [min_intervention_N, max_intervention_N]),
          ('O', [min_intervention_O, max_intervention_O]),
          ('C', [min_intervention_C, max_intervention_C]),
          ('T', [min_intervention_T, max_intervention_T]),
          ('D', [min_intervention_D, max_intervention_D]),
          #('X', [min_intervention_X, max_intervention_X]),
          #('L', [min_intervention_L, max_intervention_L]),
          #('P', [min_intervention_P, max_intervention_P]),
          #('S', [min_intervention_S, max_intervention_S]),
          #('E', [min_intervention_E, max_intervention_E]),
        ])
        return dict_ranges
    
    def refit_models(self, observational_samples):
        Y = np.asarray(observational_samples['Y'])[:,np.newaxis]
        N = np.asarray(observational_samples['N'])[:,np.newaxis]
        X = np.asarray(observational_samples['X'])[:,np.newaxis]
        T = np.asarray(observational_samples['T'])[:,np.newaxis]
        D = np.asarray(observational_samples['D'])[:,np.newaxis]
        P = np.asarray(observational_samples['P'])[:,np.newaxis]
        O = np.asarray(observational_samples['O'])[:,np.newaxis]
        S = np.asarray(observational_samples['S'])[:,np.newaxis]
        L = np.asarray(observational_samples['L'])[:,np.newaxis]
        E = np.asarray(observational_samples['E'])[:,np.newaxis]
        C = np.asarray(observational_samples['C'])[:,np.newaxis]


        functions = {}
        inputs_list = [N, np.hstack((O,S, T,D,E)), np.hstack((C,N, L,E)), np.hstack((T,S)),
                        np.hstack((D,S)), np.hstack((N,O,S, T,D,E)), np.hstack((N,T,S)),
                        np.hstack((N,D,S)), np.hstack((O,C,N, L, E, S, T, D)),
                        np.hstack((T,C,S,E,L,N)), np.hstack((T,D,S)), 
                        np.hstack((C,D,S, E, L, N)), np.hstack((N,C,T, S, N, L, E)),
                        np.hstack((N,T,D, S)), np.hstack((C,T,D, S, N, L, E))]

        output_list = [Y, Y ,  Y , Y, Y ,Y , Y,  Y , Y,Y, Y,  Y, Y, Y, Y]

        name_list = ['gp_N', 'gp_O_S_T_D_E', 'gp_C_N_L_E', 'gp_T_S', 'gp_D_S', 'gp_N_O_S_T_D_E', 'gp_N_T_S', 'gp_N_D_S', 'gp_O_C_N_L_E_S_T_D',
                    'gp_T_C_S_E_L_N', 'gp_T_D_S', 'gp_C_D_S_E_L_N', 'gp_N_C_T_S_N_L_E', 'gp_N_T_D_S', 'gp_C_T_D_S_N_L_E']
        
        parameter_list = [[1.,1.,10., False], [1.,1.,1., True], [1.,1.,1., True],[1.,1.,1., True],[1.,1.,10., True], 
                        [1.,1.,1., False], [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False], [1.,1.,1., False],
                        [1.,1.,1., False],[1.,1.,1., False], [1.,1.,1., False]]

        ## Fit all conditional models
        # NOE observationGP，musigma（0+RBF kernel）；
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_single_GP_model(X, Y, parameter_list[i])
  
        return functions
    def get_all_do(self):
        do_dict = {}
        do_dict['compute_do_N'] = compute_do_N
        do_dict['compute_do_O'] = compute_do_O
        do_dict['compute_do_C'] = compute_do_C
        do_dict['compute_do_T'] = compute_do_T
        do_dict['compute_do_D'] = compute_do_D

        do_dict['compute_do_NO'] = compute_do_NO
        do_dict['compute_do_NC'] = compute_do_NC
        do_dict['compute_do_NT'] = compute_do_NT
        do_dict['compute_do_ND'] = compute_do_ND
        do_dict['compute_do_OC'] = compute_do_OC
        do_dict['compute_do_OT'] = compute_do_OT
        do_dict['compute_do_OD'] = compute_do_OD
        do_dict['compute_do_TC'] = compute_do_TC
        do_dict['compute_do_TD'] = compute_do_TD
        do_dict['compute_do_CD'] = compute_do_CD

        do_dict['compute_do_NOC'] = compute_do_NOC
        do_dict['compute_do_NOT'] = compute_do_NOT
        do_dict['compute_do_NOD'] = compute_do_NOD
        do_dict['compute_do_NCT'] = compute_do_NCT
        do_dict['compute_do_NCD'] = compute_do_NCD
        do_dict['compute_do_NTD'] = compute_do_NTD
        do_dict['compute_do_OCT'] = compute_do_OCT
        do_dict['compute_do_OCD'] = compute_do_OCD
        do_dict['compute_do_CTD'] = compute_do_CTD
        do_dict['compute_do_OTD'] = compute_do_OTD


        return do_dict