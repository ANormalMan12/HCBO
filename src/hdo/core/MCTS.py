from .RandEmbed import random_embedding, random_projection, generate_random_matrix
from .BayeOpt import next_point_bo, init_points_dataset_bo, update_dataset_ucb
from matplotlib import pyplot as plt
import torch
import random
from zoopt import Dimension, Objective, solution, Parameter
import numpy as np
from .Racos import SRacos
from .CMAES import cmaes


class Node:
    # def __init__(self, parent, Cp, data, bounds, rand_mat, obj_func, height, UCT_type, opt_method, max_data, seed, d,
    #              is_deepest, remain_budget, pop_size):
    def __init__(self, parent, Cp, height, UCT_type, theta, seed, data, rand_mat, obj_func, d, bounds, opt_method,
                 opt_bounds, max_data, is_deepest, remain_budget, pop_size, optimizer, sigma):
        self.parent = parent
        self.lchild = None
        self.rchild = None
        self.bounds = bounds
        self.opt_bounds = opt_bounds
        self.Cp = Cp
        self.is_visit = 1
        self.rand_mat = rand_mat
        # self.rand_mat_inv = torch.pinverse(rand_mat)
        self.obj_func = obj_func
        self.height = height
        self.UCT_type = UCT_type
        self.data = data
        self.d = d
        self.theta = theta
        np.random.seed(seed)
        if optimizer == None:
            if opt_method == 'racos':
                data_nums = data['f'].shape[0]
                init_dataset = [solution.Solution(data['y'][i].tolist(), -data['f'][i].numpy().astype(np.float64)[0])
                                for i in range(data_nums)]
                dim = Dimension(self.d, self.opt_bounds.tolist(), [True] * self.d)
                obj = Objective(obj_func, dim)
                if not is_deepest:
                    para = Parameter(budget=max_data, init_samples=init_dataset, seed=seed)
                else:
                    para = Parameter(budget=remain_budget, init_samples=init_dataset, seed=seed)
                # print(para.get_budget())
                if self.d <= 100:
                    self.ub = 1
                elif self.d <= 1000:
                    self.ub = 2
                else:
                    self.ub = 3
                self.optimizer = SRacos(obj, para)
            elif opt_method == 'cmaes':
                # print(self.d, pop_size, seed)
                self.optimizer = cmaes(self.d, pop_size, self.opt_bounds.numpy(), sigma, seed)
                # self.optimizer = cmaes(pop_size, self.opt_bounds.numpy(), remain_budget, self.data, sigma, seed)
                if self.data:
                    self.optimizer.update([point.numpy() for point in self.data['y']], [float(f) for f in self.data['f']])
            elif opt_method == 'bo':
                self.optimizer = None
        else:
            if self.d <= 100:
                self.ub = 1
            elif self.d <= 1000:
                self.ub = 2
            else:
                self.ub = 3
            self.optimizer = optimizer


    def is_leaf(self):
        if not self.lchild and not self.rchild:
            return True
        else:
            return False

    def visit(self):
        self.is_visit += 1

    # def splitable(self):
    #     if self.parent == None:
    #         return True
    #     elif self.parent.bounds == self.bounds:
    #         return False

    def update_cp(self, Cp):
        self.Cp = Cp

    def UCT(self):
        if self.UCT_type == 'mean':
            # return torch.mean(self.data['f'], dim=0) + \
            #        self.Cp * torch.sqrt(torch.log(torch.tensor(self.parent.is_visit)) / self.is_visit)
            return self.data['f'].mean(dim=0) + \
                   self.Cp * torch.sqrt(torch.log(torch.tensor(self.parent.is_visit)) / self.is_visit)\
                   * torch.sqrt(torch.tensor(8)) if self.data else 1e10
        elif self.UCT_type == 'max':
            return self.data['f'].max() + \
                   self.Cp * torch.sqrt(torch.log(torch.tensor(self.parent.is_visit)) / self.is_visit) \
                   if self.data else 1e10
        else:
            print('Please choose the supported UCT calculation method. '
                  'We use the default type (mean) to continue the program.')
            return self.data['f'].mean(dim=0) + \
                   self.Cp * torch.sqrt(torch.log(torch.tensor(self.parent.is_visit)) / self.is_visit) \
                   * torch.sqrt(torch.tensor(8)) if self.data else 1e10


class MCTS:
    def __init__(self, Cp, cp_decay_rate, split_threshold, split_type, UCT_type, max_height, budget, init_nums,
                 init_data, rand_mat, bounds, obj_func, D, d, opt_bounds, seed, opt_method, kernel_type, pop_size, path, sigma):
    # def __init__(self, Cp, split_threshold, split_type, budget, init_nums, init_data, max_height, rand_mat, bounds,
    #              obj_func, D, d, seed, opt_method, UCT_type, cp_decay_rate, kernel_type, pop_size):
        torch.manual_seed(seed)
        self.Cp = Cp
        self.cp_decay_rate = cp_decay_rate
        self.theta = split_threshold
        self.split_type = split_type
        # self.positive_percentage = positive_percentage
        self.budget = budget
        self.init_nums = init_nums
        self.UCT_type = UCT_type
        self.pop_size = pop_size
        self.max_height = max_height
        self.bounds = bounds
        self.obj_func = obj_func
        self.D = D
        self.d = d
        self.opt_bounds = opt_bounds
        self.opt_method = opt_method
        self.seed = seed
        self.kernel_type = kernel_type
        self.sigma = sigma
        # self.root = Node(None, self.Cp, 1, self.UCT_type, self.theta, self.seed, init_data, rand_mat, self.obj_func, self.d,
        #                  self.bounds, self.opt_method, self.opt_bounds, self.theta, 1 == self.max_height,
        #                  self.budget - init_data['f'].shape[0], self.pop_size, None, self.sigma)
        # # self.root = Node(None, Cp, init_data, bounds, rand_mat, obj_func, 1, self.UCT_type, opt_method, split_threshold,
        # #                  seed, d, 1 == max_height, self.budget - init_data['f'].shape[0], self.pop_size)
        # self.total_data = {'x': init_data['x'].clone(), 'f': init_data['f'].clone()}
        # n = torch.argmax(self.total_data['f'])
        # self.cur_best_x = self.total_data['x'][n]
        # self.cur_best_f = self.total_data['f'][n]

        if self.init_nums == 0:
            self.root = Node(None, self.Cp, 1, self.UCT_type, self.theta, self.seed, None, rand_mat, self.obj_func,
                             self.d, self.bounds, self.opt_method, self.opt_bounds, self.theta, 1 == self.max_height,
                             self.budget, self.pop_size, None, self.sigma)
            self.total_data = None
            self.cur_best_x = None
            self.cur_best_f = -1e10
        else:
            self.root = Node(None, self.Cp, 1, self.UCT_type, self.theta, self.seed, init_data, rand_mat, self.obj_func,
                             self.d, self.bounds, self.opt_method, self.opt_bounds, self.theta, 1 == self.max_height,
                             self.budget - init_data['f'].shape[0], self.pop_size, None, self.sigma)
            self.total_data = {'x': init_data['x'].clone(), 'f': init_data['f'].clone()}
            n = torch.argmax(self.total_data['f'])
            self.cur_best_x = self.total_data['x'][n]
            self.cur_best_f = self.total_data['f'][n]
        self.cur_best_bounds = self.bounds
        self.nodes = [self.root]
        self.leaves = [self.root]
        self.path = path
        # self.nodes = []
        # self.leaves = [self.root]

        if self.init_nums:
            self.split_tree()
        print('MCTS tree has been initialized.')
        print('Current best f(x): ', self.cur_best_f)
        if self.D <= 1e2:
            print('Current best x: ', self.cur_best_x)

    def select_dim(self, bounds, data, D):
        diff = torch.zeros(D)
        for i in range(D):
            split_line = torch.mean(bounds[i])
            f1 = torch.index_select(data['f'], dim=0,
                                    index=torch.nonzero(data['x'][:, i] <= split_line).reshape(1, -1).squeeze())
            f2 = torch.index_select(data['f'], dim=0,
                                    index=torch.nonzero(data['x'][:, i] > split_line).reshape(1, -1).squeeze())
            diff[i] = torch.abs(f1.mean(dim=0)-f2.mean(dim=0))
        return torch.argmax(diff)

    def select_bounds(self, data, select_type, D, bounds):
        if select_type == 'mean':
            threshold = data['f'].mean(dim=0)
            chosen_data = torch.index_select(data['x'], dim=0, index=torch.nonzero(data['f'] >= threshold)[:, 0])
            bound = torch.ones(D, 2)
            bound[:, 0] = chosen_data.min(dim=0).values
            bound[:, 1] = chosen_data.max(dim=0).values
        elif select_type[0:3] == 'top':
            topk = int(int(select_type[3:-1]) / 100 * data['f'].shape[0])
            chosen_data = \
                torch.index_select(data['x'], dim=0,
                                   index=torch.nonzero(data['f'] >= torch.topk(data['f'].reshape(1, -1)[0], topk).values[-1])[:, 0])
            # print(torch.topk(data['f'].reshape(1, -1)[0], topk).values[-1], data['f'])
            # print(topk, chosen_data.shape)
            bound = torch.ones(D, 2)
            bound[:, 0] = chosen_data.min(dim=0).values
            bound[:, 1] = chosen_data.max(dim=0).values
        elif select_type[0:5] == 'bound':
            # print(bounds)
            new_range = int(select_type[5:-1]) / 100 * (bounds.t()[1] - bounds.t()[0])
            n = torch.argmax(data['f'])
            best_x = data['x'][n]
            # print(best_x)
            bound = torch.ones(D, 2)
            bound_high = torch.cat([best_x + new_range / 2, bounds.t()[1]]).reshape(2, -1)
            bound[:, 1] = bound_high.min(dim=0).values
            bound_low = torch.cat([best_x - new_range / 2, bounds.t()[0]]).reshape(2, -1)
            bound[:, 0] = bound_low.max(dim=0).values
            # print(bound)
        else:
            print('Please choose the supported split domain method. '
                  'We use the default type (mean) to continue the program.')
            threshold = data['f'].mean(dim=0)
            chosen_data = torch.index_select(data['x'], dim=0, index=torch.nonzero(data['f'] >= threshold)[:, 0])
            bound = torch.ones(D, 2)
            bound[:, 0] = chosen_data.min(dim=0).values
            bound[:, 1] = chosen_data.max(dim=0).values
        return bound

    def split_node(self, leaf):
        bounds_small = self.select_bounds(leaf.data, self.split_type, self.D, leaf.bounds)
        # bounds = leaf.bounds
        # if self.split_type == 'rand':
        #     split_dim = torch.randint(0, self.D, (1, 1))[0][0]
        # elif self.split_type == 'max':
        #     bounds_range = bounds.T[1] - bounds.T[0]
        #     bounds_range = bounds_range == bounds_range.max()
        #     dim = [i for i in range(self.D) if bounds_range[i]]
        #     split_dim = random.choice(dim)
        #     # print(bounds_range, dim, split_dim)
        #     # split_dim = torch.argmax(bounds.T[1] - bounds.T[0])
        # elif self.split_type == 'improve':
        #     split_dim = self.select_dim(bounds, leaf.data, self.D)
        # else:
        #     print('Please choose the supported split dimension method. '
        #           'We use the default type (improve) to continue the program.')
        #     split_dim = self.select_dim(bounds, leaf.data, self.D)
        # split_line = torch.mean(bounds[split_dim])
        # bounds_l = bounds.clone()
        # bounds_l[split_dim, 1] = split_line
        # bounds_r = bounds.clone()
        # bounds_r[split_dim, 0] = split_line
        # part_l, part_r = [], []
        # for i in range(leaf.data['x'].shape[0]):
        #     if leaf.data['x'][i, split_dim] <= split_line:
        #         part_l.append(i)
        #     else:
        #         part_r.append(i)
        # data_l = {'x': torch.cat([leaf.data['x'][i].unsqueeze(dim=0) for i in part_l], dim=0),
        #           'f': torch.cat([leaf.data['f'][i].unsqueeze(dim=0) for i in part_l], dim=0)} if part_l else None
        # data_r = {'x': torch.cat([leaf.data['x'][i].unsqueeze(dim=0) for i in part_r], dim=0),
        #           'f': torch.cat([leaf.data['f'][i].unsqueeze(dim=0) for i in part_r], dim=0)} if part_r else None
        # bounds_big = leaf.bounds
        bound_vector = self.opt_bounds[:, 1] ** 2
        # sigma_small = (bounds_small[:, 1] - bounds_small[:, 0]) / (2 * torch.sqrt(bound_vector.sum()))
        sigma_small = (bounds_small[:, 1] - bounds_small[:, 0]) / (2 * torch.log10(torch.tensor(self.d + 1)) * torch.sqrt(bound_vector.sum()))
        # sigma_big = (bounds_big[:, 1] - bounds_big[:, 0]) / (2 * torch.sqrt(torch.tensor(self.d)))
        rand_mat_small = generate_random_matrix(self.d, self.D, sigma_small, torch.randint(0, 10000, (1, 1))[0][0])
        # rand_mat_r = generate_random_matrix(self.d, self.D, sigma_r, torch.randint(0, 10000, (1, 1))[0][0])
        # data_small = init_points_dataset_bo(self.init_nums, rand_mat_small, self.opt_bounds, bounds_small, self.obj_func)
        # # self.save_log(data_small)
        # self.total_data = update_dataset_ucb(None, data_small['x'], data_small['f'], self.total_data)
        # # print(self.total_data['f'].shape)
        # n = torch.argmax(data_small['f'])
        # if data_small['f'][n] > self.cur_best_f:
        #     self.cur_best_f = data_small['f'][n]
        #     self.cur_best_x = data_small['x'][n]
        #     self.cur_best_bounds = bounds_small
        # # data_r = init_points_dataset_bo(self.init_nums, rand_mat_r, torch.tensor([[-1, 1]] * self.d), bounds_r,
        # #                                 self.obj_func)
        # # self.total_data = update_dataset_ucb(None, data_r['x'], data_r['f'], self.total_data)
        # # print(self.total_data['f'].shape)
        # # n = torch.argmax(data_r['f'])
        # # if data_r['f'][n] > self.cur_best_f:
        # #     self.cur_best_f = data_r['f'][n]
        # #     self.cur_best_x = data_r['x'][n]
        # leaf.lchild = Node(leaf, self.Cp, leaf.height + 1, self.UCT_type, self.theta, self.seed, data_small, rand_mat_small,
        #                    self.obj_func, self.d, bounds_small,  self.opt_method, self.opt_bounds, self.theta,
        #                    (leaf.height + 1) == self.max_height, self.budget - self.total_data['f'].shape[0],
        #                    self.pop_size, None, self.sigma)
        if self.init_nums != 0:
            data_small = init_points_dataset_bo(self.init_nums, rand_mat_small, self.opt_bounds, bounds_small, self.obj_func)
            # self.save_log(data_small)
            self.total_data = update_dataset_ucb(None, data_small['x'], data_small['f'], self.total_data)
        # print(self.total_data['f'].shape)
            n = torch.argmax(data_small['f'])
            if data_small['f'][n] > self.cur_best_f:
                self.cur_best_f = data_small['f'][n]
                self.cur_best_x = data_small['x'][n]
                self.cur_best_bounds = bounds_small
            leaf.lchild = Node(leaf, self.Cp, leaf.height + 1, self.UCT_type, self.theta, self.seed, data_small,
                               rand_mat_small, self.obj_func, self.d, bounds_small, self.opt_method, self.opt_bounds,
                               self.theta, (leaf.height + 1) == self.max_height,
                               self.budget - self.total_data['f'].shape[0], self.pop_size, None, self.sigma)
        # data_r = init_points_dataset_bo(self.init_nums, rand_mat_r, torch.tensor([[-1, 1]] * self.d), bounds_r,
        #                                 self.obj_func)
        # self.total_data = update_dataset_ucb(None, data_r['x'], data_r['f'], self.total_data)
        # print(self.total_data['f'].shape)
        # n = torch.argmax(data_r['f'])
        # if data_r['f'][n] > self.cur_best_f:
        #     self.cur_best_f = data_r['f'][n]
        #     self.cur_best_x = data_r['x'][n]
        else:
            leaf.lchild = Node(leaf, self.Cp, leaf.height + 1, self.UCT_type, self.theta, self.seed, None,
                               rand_mat_small, self.obj_func, self.d, bounds_small, self.opt_method, self.opt_bounds,
                               self.theta, (leaf.height + 1) == self.max_height,
                               self.budget - self.total_data['f'].shape[0], self.pop_size, None, self.sigma)
        # bounds_big = leaf.bounds
        # sigma_big = (bounds_big[:, 1] - bounds_big[:, 0]) / (2 * torch.sqrt(torch.tensor(self.d)))
        # rand_mat_big = generate_random_matrix(self.d, self.D, sigma_big, torch.randint(0, 10000, (1, 1))[0][0])
        # data_big = {'x': leaf.data['x'].clone(), 'f': leaf.data['f'].clone()}
        # data_big['y'] = random_projection(data_big['x'], torch.pinverse(rand_mat_big), bounds_big)

        leaf.rchild = Node(leaf, self.Cp, leaf.height + 1, self.UCT_type, leaf.theta + self.theta, self.seed, leaf.data,
                           leaf.rand_mat, self.obj_func, self.d, leaf.bounds, self.opt_method, self.opt_bounds,
                           self.theta, (leaf.height + 1) == self.max_height,
                           self.budget - self.total_data['f'].shape[0], self.pop_size, leaf.optimizer, self.sigma)
        # leaf.rchild = Node(leaf, self.Cp, leaf.height + 1, self.UCT_type, leaf.theta + self.theta, self.seed, data_big,
        #                    rand_mat_big, self.obj_func, self.d, bounds_big, self.opt_method, self.theta,
        #                    (leaf.height + 1) == self.max_height, self.budget - self.total_data['f'].shape[0],
        #                    self.pop_size, leaf.optimizer)
        leaf.rchild.is_visit = leaf.is_visit
        # leaf.lchild = Node(leaf, self.Cp, data_l, bounds_l, rand_mat_l, self.obj_func, leaf.height + 1, self.UCT_type,
        #                    self.opt_method, self.theta, self.seed, self.d, (leaf.height + 1) == self.max_height,
        #                    self.budget - self.total_data['f'].shape[0], self.pop_size)
        # leaf.rchild = Node(leaf, self.Cp, data_r, bounds_r, rand_mat_r, self.obj_func, leaf.height + 1, self.UCT_type,
        #                    self.opt_method, self.theta, self.seed, self.d, (leaf.height + 1) == self.max_height,
        #                    self.budget - self.total_data['f'].shape[0], self.pop_size)

        self.nodes.append(leaf.lchild)
        self.nodes.append(leaf.rchild)
        self.leaves = [node for node in self.nodes if node.is_leaf()]
        # print(self.nodes)
        return leaf.lchild, leaf.rchild

    def split_tree(self):
        flag = True
        while flag:
            split_num = 0
            for leaf in self.leaves:
                # print(self.leaves)
                # print(leaf.data['f'] if leaf.data else 'None')
                # print(self.total_data['f'])
                # if leaf.data['f'].shape[0] >= self.theta and leaf.height < self.max_height \
                #         and not leaf.lchild and not leaf.rchild:

                if leaf.data and leaf.data['f'].shape[0] >= leaf.theta and leaf.height < self.max_height:
                    a, b = self.split_node(leaf)
                    # print('child', a.data, b.data)
                    split_num += 1
            if split_num == 0:
                flag = False

    def back_propagation(self, leaf):
        node_now = leaf.parent
        while node_now:
            node_now.visit()
            node_now = node_now.parent

    def plot_bounds(self, bounds):
        if self.D == 2:
            plt.plot([bounds[0, 0], bounds[0, 1], bounds[0, 1], bounds[0, 0], bounds[0, 0]],
                     [bounds[1, 0], bounds[1, 0], bounds[1, 1], bounds[1, 1], bounds[1, 0]], linewidth=2.5)
            # plt.axvline(bounds[0, 0])
            # plt.axvline(bounds[0, 1])
            # plt.axhline(bounds[1, 0])
            # plt.axhline(bounds[1, 1])
            # plt.scatter(32.768 / 2, 32.768 / 2)
            plt.axvline(32.768 / 2, color='red')
            plt.xlim(self.bounds[0, 0], self.bounds[0, 1])
            plt.ylim(self.bounds[1, 0], self.bounds[1, 1])
            # plt.savefig(fname='visualize/' + str(self.total_data['f'].shape[0]) + '.png', dpi=1200)
            plt.show()

    def save_log(self, message):
        file = open(str(self.path + '/log_seed' + str(self.seed) + '.txt'), 'a')
        file.write(str(message) + '\n')
        file.close()

    def save_tree_info(self):
        file = open(str(self.path + '/log_seed' + str(self.seed) + '.txt'), 'a')
        for node in self.nodes:
            if node.data:
                file.write('height:' + str(node.height) + '\t is_visit:' + str(node.is_visit) +
                           '\t data number:' + str(node.data['f'].shape[0]) + '\t theta:' + str(node.theta) +
                           '\t best data:' + str(node.data['f'].max()) + '\t is leaf:' + str(node.is_leaf()) +
                           '\t optimizer:' + str(node.optimizer) + '\n')
            file.write('bounds:\n')
            file.write(str(node.bounds) + '\n')
            file.write('rand_mat:\n')
            file.write(str(node.rand_mat) + '\n')
        file.close()

    def search(self):
        while not self.total_data or self.budget > self.total_data['f'].shape[0]:
            if self.opt_method == 'cmaes':
                print('Iteration : ', self.total_data['f'].shape[0] // self.pop_size if self.total_data else 0)
            else:
                print('Iteration : ', self.total_data['f'].shape[0])
            if self.root.is_leaf():
                cur_node = self.root
            else:
                for node in self.nodes:
                    node.update_cp(self.Cp / (1 + self.cp_decay_rate * self.total_data['f'].shape[0]))
                cur_node = self.leaves[torch.argmax(torch.tensor([leaf.UCT() for leaf in self.leaves]))]
                # print([(node.UCT(), node.height, node.data['f'].max(), node.is_visit, node.parent.is_visit) for node in self.nodes if node != self.root])
                self.save_log([(node.UCT(), node.height, node.data['f'].max() if node.data else 0, node.is_visit, node.parent.is_visit) for node in self.nodes if node != self.root])
                # cur_node = self.nodes[torch.argmax(torch.tensor([node.UCT() for node in self.nodes]))]
            cur_node.visit()
            # print(cur_node.height, cur_node.data['f'].shape[0], cur_node.bounds.t())
            self.back_propagation(cur_node)
            if self.opt_method == 'bo':
                is_sampled, next_y = \
                    next_point_bo(cur_node.data, 0.2 * self.d * torch.log(torch.tensor(2 * cur_node.data['f'].shape[0])),
                                  self.opt_bounds, self.kernel_type)
            elif self.opt_method == 'racos':
                # is_sampled, next_y = next_point_racos(cur_node.data, torch.tensor([[-1, 1.]] * self.d),
                #                                       self.obj_func, self.seed, self.positive_percentage)
                is_sampled, solution, next_y = cur_node.optimizer.gen_next_point(cur_node.ub)
            elif self.opt_method == 'cmaes':
                is_sampled, points, data = cur_node.optimizer.gen_next_point()
            else:
                is_sampled, next_y = None, None
                print('Please select the supported method.')
            # is_sampled, data = self.sample()
            if is_sampled:
                if self.opt_method == 'cmaes':
                    next_x = [random_embedding(next_y, cur_node.rand_mat, cur_node.bounds) for next_y in data]
                    next_x = torch.cat(next_x, dim=0)
                    next_y = torch.cat(data, dim=0)
                    print(next_y.max(dim=1).values)
                    print(next_y.min(dim=1).values)
                    next_f = self.obj_func(next_x).reshape(-1, 1)
                    # print(next_f)
                    cur_node.data = update_dataset_ucb(next_y, next_x, next_f, cur_node.data)
                    self.total_data = update_dataset_ucb(None, next_x, next_f, self.total_data)
                    cur_node.optimizer.update(points, [-float(f.clone()) for f in next_f])
                    n = torch.argmax(next_f)
                    if next_f[n] > self.cur_best_f:
                        self.cur_best_f = next_f[n]
                        self.cur_best_x = next_x[n]
                        self.cur_best_bounds = cur_node.bounds
                    if self.D <= 1e2:
                        print('Current best x: ', self.cur_best_x)
                    print('Current best f(x): ', self.cur_best_f)
                    self.split_tree()
                else:
                    next_x = random_embedding(next_y, cur_node.rand_mat, cur_node.bounds)
                    next_f = self.obj_func(next_x)
                    # print(next_y)
                    cur_node.data = update_dataset_ucb(next_y, next_x, next_f, cur_node.data)
                    # print(self.total_data['f'].shape[0])
                    # if cur_node != self.root:
                    self.total_data = update_dataset_ucb(None, next_x, next_f, self.total_data)
                    if self.opt_method == 'racos':
                        cur_node.optimizer.update(solution, -next_f.clone().numpy().astype(np.float64)[0])
                    # print(self.total_data['f'].shape[0])
                    # self.budget -= 1
                    if next_f > self.cur_best_f:
                        self.cur_best_f = next_f
                        self.cur_best_x = next_x
                        self.cur_best_bounds = cur_node.bounds
                    if self.D <= 1e3:
                        print('Next x: ', next_x)
                    print('Next f(x): ', next_f)
                    self.split_tree()
            else:
                print('MCTS early stopped.')
                print('Final best f(x): ', self.cur_best_f)
                if self.D <= 1e2:
                    print('Final best x: ', self.cur_best_x)
                self.save_tree_info()
                return False, cur_node.bounds
        print('MCTS completed.')
        print('Final best f(x): ', self.cur_best_f)
        if self.D <= 1e2:
            print('Final best x: ', self.cur_best_x)
        self.save_tree_info()
        return True, self.cur_best_bounds






