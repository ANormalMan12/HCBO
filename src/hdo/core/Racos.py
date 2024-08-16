import numpy as np
from zoopt.utils.tool_function import ToolFunction
from zoopt.algos.opt_algorithms.racos.racos_classification import RacosClassification
from zoopt.algos.opt_algorithms.racos.racos_common import RacosCommon
import torch
from zoopt.solution import Solution
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt
from .ObjFunc import ackley, sphere


class SRacos(RacosCommon):
    """
    The class SRacos represents Sequential Racos algorithm. It's inherited from RacosCommon.
    """

    def __init__(self, objective, parameter):
        """
        Initialization.
        """
        RacosCommon.__init__(self)
        self.clear()
        self.set_objective(objective)
        self.set_parameters(parameter)
        self.init_attribute()
        self.max_distinct_repeat_times = 100
        self.current_not_distinct_times = 0
        return

    def gen_next_point(self, ub=1):
        """
        SRacos optimization.

        :param objective: an Objective object
        :param parameter: a Parameter object
        :param strategy: replace strategy
        :param ub: uncertain bits, which is a parameter of SRacos
        :return: Recommended point
        """
        sampled_data = self._positive_data + self._negative_data
        if np.random.random() < self._parameter.get_probability():
            classifier = RacosClassification(
                self._objective.get_dim(), self._positive_data, self._negative_data, ub)
            classifier.mixed_classification()
            solution, distinct_flag = self.distinct_sample_classifier(
                classifier, sampled_data, True, self._parameter.get_train_size())
        else:
            solution, distinct_flag = self.distinct_sample(self._objective.get_dim(), sampled_data)
        # panic stop
        if solution is None:
            ToolFunction.log(" [break loop] because solution is None")
            return False, self._best_solution
        if distinct_flag is False:
            self.current_not_distinct_times += 1
            if self.current_not_distinct_times >= self.max_distinct_repeat_times:
                ToolFunction.log(
                    "[break loop] because distinct_flag is false too much times")
                return False, self._best_solution
        return True, solution, torch.tensor([solution.get_x()])

    def update(self, solution, value, strategy='WR'):
        solution.set_value(value)
        bad_ele = self.replace(self._positive_data, solution, 'pos')
        self.replace(self._negative_data, bad_ele, 'neg', strategy)
        self._best_solution = self._positive_data[0]

    def replace(self, iset, x, iset_type, strategy='WR'):
        """
        Replace a solution(chosen by strategy) in iset with x.

        :param iset: a solution list
        :param x: a Solution object
        :param iset_type: 'pos' or 'neg'
        :param strategy: 'WR': worst replace or 'RR': random replace or 'LM': replace the farthest solution
        :return: the replaced solution
        """
        if strategy == 'WR':
            return self.strategy_wr(iset, x, iset_type)
        elif strategy == 'RR':
            return self.strategy_rr(iset, x)
        elif strategy == 'LM':
            best_sol = min(iset, key=lambda x: x.get_value())
            return self.strategy_lm(iset, best_sol, x)

    def binary_search(self, iset, x, begin, end):
        """
        Find the first element larger than x.

        :param iset: a solution set
        :param x: a Solution object
        :param begin: begin position
        :param end: end position
        :return: the index of the first element larger than x
        """
        x_value = x.get_value()
        if x_value <= iset[begin].get_value():
            return begin
        if x_value >= iset[end].get_value():
            return end + 1
        if end == begin + 1:
            return end
        mid = begin + (end - begin) // 2
        if x_value <= iset[mid].get_value():
            return self.binary_search(iset, x, begin, mid)
        else:
            return self.binary_search(iset, x, mid, end)

    def strategy_wr(self, iset, x, iset_type):
        """
        Replace the worst solution in iset.

        :param iset: a solution set
        :param x: a Solution object
        :param iset_type: 'pos' or 'neg'
        :return: the worst solution
        """
        if iset_type == 'pos':
            index = self.binary_search(iset, x, 0, len(iset) - 1)
            iset.insert(index, x)
            worst_ele = iset.pop()
        else:
            worst_ele, worst_index = Solution.find_maximum(iset)
            if worst_ele.get_value() > x.get_value():
                iset[worst_index] = x
            else:
                worst_ele = x
        return worst_ele

    def strategy_rr(self, iset, x):
        """
        Replace a random solution in iset.

        :param iset: a solution set
        :param x: a Solution object
        :return: the replaced solution
        """
        len_iset = len(iset)
        replace_index = np.random.randint(0, len_iset)
        replace_ele = iset[replace_index]
        iset[replace_index] = x
        return replace_ele

    #
    def strategy_lm(self, iset, best_sol, x):
        """
        Replace the farthest solution from best_sol

        :param iset: a solution set
        :param best_sol: the best solution, distance between solution in iset and best_sol will be computed
        :param x: a Solution object
        :return: the farthest solution (has the largest margin) in iset
        """
        farthest_dis = 0
        farthest_index = 0
        for i in range(len(iset)):
            dis = self.distance(iset[i].get_x(), best_sol.get_x())
            if dis > farthest_dis:
                farthest_dis = dis
                farthest_index = i
        farthest_ele = iset[farthest_index]
        iset[farthest_index] = x
        return farthest_ele

    @staticmethod
    def distance(x, y):
        """
        Get the distance between the list x and y
        :param x: a list
        :param y: a list
        :return: Euclidean distance
        """
        dis = 0
        for i in range(len(x)):
            dis += (x[i] - y[i])**2
        return np.sqrt(dis)


if __name__ == "__main__":

    from core import RandEmbed

    sigma = [7.3271] * 100

    # rand_mat = RandEmbed.generate_random_matrix(20, 100, sigma, 0)
    #
    #
    # def _ackley(solution):
    #     x = solution.get_x()
    #     x = torch.tensor([x])
    #     # print(x)
    #     # x = RandEmbed.random_embedding(x, rand_mat, torch.tensor([[-32.768, 32.768]] * 100))
    #     res = -ackley(x)
    #     # bias = 0.2
    #     # value = -20 * np.exp(-0.2 * np.sqrt(sum([(i - bias) * (i - bias) for i in x]) / len(x))) - \
    #     #         np.exp(sum([np.cos(2.0*np.pi*(i-bias)) for i in x]) / len(x)) + 20.0 + np.e
    #     return float(res)
    # dim_size = 20  # dimension
    # total_data = torch.zeros(20, 10000)
    # dim = Dimension(dim_size, [[-1, 1]] * dim_size, [True] * dim_size)  # or dim = Dimension2([(ValueType.CONTINUOUS, [-1, 1], 1e-6)]*dim_size)
    # obj = Objective(_ackley, dim)
    # perform optimization
    # solution = SRacos(obj, Parameter(budget=10000))
    # for i in range(10000):
    #     print(i)
    #     _, y, _ = solution.gen_next_point(ub=1)
    #     f = _ackley(y)
    #     solution.update(y, f)
    # print(solution._best_solution.get_x())
    # print(solution._best_solution.get_value())
    # solution_list = ExpOpt.min(obj, Parameter(budget=100), repeat=3, plot=True, plot_file="progress.png")
    # for solution in solution_list:
    #     print(solution.get_x(), solution.get_value())
    total_data = torch.zeros(20, 1000)
    for i in range(20):
        # rand_mat = RandEmbed.generate_random_matrix(20, 100, sigma, i)

        def _ackley(solution):
            x = solution.get_x()
            x = torch.tensor([x])
            # print(x)
            # x = RandEmbed.random_embedding(x, rand_mat, torch.tensor([[-32.768, 32.768]] * 100))
            res = -ackley(x)
            # bias = 0.2
            # value = -20 * np.exp(-0.2 * np.sqrt(sum([(i - bias) * (i - bias) for i in x]) / len(x))) - \
            #         np.exp(sum([np.cos(2.0*np.pi*(i-bias)) for i in x]) / len(x)) + 20.0 + np.e
            return float(res)

        dim_size = 10 # dimension
        dim = Dimension(dim_size, [[-32.768, 32.768]] * dim_size, [True] * dim_size)  # or dim = Dimension2([(ValueType.CONTINUOUS, [-1, 1], 1e-6)]*dim_size)
        obj = Objective(_ackley, dim)
        solution = Opt.min(obj, Parameter(budget=1000))#, reducedim=True, low_dimension=Dimension(20, [[-1, 1]] * 20, [True] * 20)))
        # print(solution.get_value())
        res = obj.get_history()
        total_data[i] = torch.tensor(res)
        obj.clean_history()

    # print the solution
    # total_data[i] = torch.tensor(obj.get_history())
    torch.save(total_data, '../results/ackley/racos_test_10.pt')
    print(total_data)
    print(total_data.shape)



