"""Particle Swarm Optimization"""
import numpy as np
from matplotlib import pyplot as plt
from utils import Particle
from utils import print_params


class PSO:
    def __init__(self, objective, part_num, num_itr, var_size, net=None):
        """
        Particle Swarm Optimization

        :param objective: cost function as an objective
        :param part_num: integer, number of particles
        :param num_itr: integer, number of iterations
        :param var_size: list, upper and lower bounds of each parameter,
                        as in [[x1_min,x1_max], [x2_min,x2_max],..., [xn_min,xn_max]]
        """
        self.part_num = part_num  # Number of the particles
        self.dim = len(var_size)  # Dimension of the particle
        self.num_itr = num_itr  # Run how many iterations
        self.objective = objective  # Objective function to be optimize
        self.w = 0.6  # initial weight
        self.c1 = 1.49
        self.c2 = 1.49
        self.var_size = var_size  # Length must correspond to the dimension of particle
        self.vmax = 0.5  # Maximum search velocity
        self.vmin = 0.01  # Minimum search velocity
        self.GlobalBest_Cost = 1e5
        self.GlobalBest_Pos = []
        # Array to hold Best costs on each iterations
        self.Best_Cost = []
        # Save space for particles
        self.particle = []
        assert self.dim == len(self.var_size)
        self.net = net

    def init_population(self):
        """
        Initialize all the particles and find the temporary best parameter.

        :return: None
        """
        print('Initializing...')
        for i in range(self.part_num):
            x = Particle()
            # initialize random position
            x.Pos = np.zeros(self.dim)
            for j in range(len(x.Pos)):
                x.Pos[j] = np.random.uniform(self.var_size[j][0], self.var_size[j][1])
            # calculate cost from random parameters
            x.Cost = self.objective(x.Pos)
            x.Vel = np.zeros(self.dim)
            x.Best_pos = x.Pos
            x.Best_cost = x.Cost
            self.particle.append(x)

            if self.particle[i].Best_cost < self.GlobalBest_Cost:
                self.GlobalBest_Cost = self.particle[i].Best_cost
                self.Best_Cost.append(self.GlobalBest_Cost)
                self.GlobalBest_Pos = self.particle[i].Best_pos
        print('Initialize complete, with best cost =', self.GlobalBest_Cost)

    def iterator(self):
        """
        Run the iterations to find the best parameters.

        :return: None
        """
        print('Iterator running...')
        for i in range(self.num_itr):
            for j in range(self.part_num):
                # create r1,r2
                r1 = np.random.uniform(self.vmin, self.vmax, self.dim)
                r2 = np.random.uniform(self.vmin, self.vmax, self.dim)
                # Update
                self.particle[j].Vel = self.w * self.particle[j].Vel \
                                       + self.c1 * r1 * (self.particle[j].Best_pos - self.particle[j].Pos) \
                                       + self.c2 * r2 * (self.GlobalBest_Pos - self.particle[j].Pos)
                self.particle[j].Pos = self.particle[j].Pos + self.particle[j].Vel
                # Check whether position out of search space
                for x in range(len(self.particle[j].Pos)):
                    if self.particle[j].Pos[x] > self.var_size[x][1]:
                        self.particle[j].Pos[x] = self.var_size[x][1]
                    if self.particle[j].Pos[x] < self.var_size[x][0]:
                        self.particle[j].Pos[x] = self.var_size[x][0]
                    assert self.var_size[x][1] >= self.particle[j].Pos[x] >= self.var_size[x][0]
                # self.particle[j].Pos[2] = int(self.particle[j].Pos[2])
                # Recalculate cost
                self.particle[j].Cost = self.objective(self.particle[j].Pos)
                if self.particle[j].Cost < self.particle[j].Best_cost:
                    self.particle[j].Best_cost = self.particle[j].Cost
                    self.particle[j].Best_pos = self.particle[j].Pos
                    if self.particle[j].Best_cost < self.GlobalBest_Cost:
                        self.GlobalBest_Cost = self.particle[j].Best_cost
                        self.GlobalBest_Pos = self.particle[j].Best_pos
            self.Best_Cost.append(self.GlobalBest_Cost)
            self.w = self.w * 0.99
            print('iteration', i + 1, ': Cost=', self.GlobalBest_Cost)
            print_params(self.GlobalBest_Pos, net=self.net)

    def plot_curve(self):
        """
        Plot optimizer curve

        :return: None
        """
        plt.plot(self.Best_Cost)
        plt.ylabel("Objective costs")
        plt.xlabel("iteration number")

    def run(self):
        """
        General call for the whole optimization.

        :return: None
        """
        print('PSO start running...')
        self.init_population()
        self.iterator()
        print("Iteration completed.")
        self.plot_curve()
        print_params(self.GlobalBest_Pos, net=self.net)

