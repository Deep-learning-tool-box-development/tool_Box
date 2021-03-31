#!/usr/bin/env python
# coding: utf-8

""" SA optimizer """
import numpy as np
import math


class SA():

    def __init__(self, objective, initial_temp, final_temp, alpha, var_size):
        """
        :param objective: cost function as an objective
        :param initial_temp: double, manually set initial_temp, e.g. 90
        :param final_temp: double, stop_temp, e.g. 0.1
        :param alpha: double, temperature changing step
        :param var_size: list, upper and lower bounds of each parameter
        :param num_itr: int, iteration times
        """
        self.interval = (0, 1)  # set a range (0,1)
        self.objective = objective  # Objective network to be optimize
        self.initial_temp = initial_temp  # 90
        self.final_temp = final_temp  # .1
        self.alpha = alpha  # 0.01 衰减因子
        self.var_size = var_size  # [[],[],[]]
        self.dim = np.zeros(len(var_size))
       # self.num_itr = num_itr

    def run(self):
        """
        :return: current state, current cost, state lists, cost lists
        """
        """ Optimize object network with the simulated annealing algorithm."""
        #for i in range(self.num_itr):
        initial_state = self.random_start()  # start from a random state, multiple dimension
        current_temp = self.initial_temp
        current_state = initial_state
        cost = self.objective(current_state)
        solution = current_state
        states, costs = [current_state], [cost]
        while current_temp > self.final_temp:
            # print("current_state", current_state)
            neighbour_of_current = self.random_neighbour(current_state)
            # print("now_neighbor", neighbour_of_current)
            # Check if neighbor is best so far
            print("out_current_state", self.objective(current_state), "out_neighbour", self.objective(neighbour_of_current))
            cost_diff = self.objective(current_state) - self.objective(neighbour_of_current)
            # print("cost_diff", cost_diff)

            # if the new solution is better, accept it
            if cost_diff > 0:
                solution = neighbour_of_current
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if np.random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                    solution = neighbour_of_current
                # decrement the temperature
            states.append(current_state)
            current_temp -= self.alpha
            costs.append(self.objective(solution))
        print(#'iteration', i + 1, ': Cost=', self.objective(solution),
                'Best parameters: ',
              '\ndropout=', solution[0],
              'learning rate=', solution[1],
              'batch size=', int(solution[2]))

    def random_start(self):
        """ Random point in the interval """
        print("___START____")
        rd_state = self.dim
        for i in range(len(self.dim)):
            #print("rd_point", rd_state)
            rd_point = np.random.uniform(self.var_size[i][0], self.var_size[i][1])
            rd_state[i] = rd_point
        print("init_random_state", rd_state)
        return rd_state

    def random_neighbour(self, state):  # fraction_origin=1
        """Find neighbour of current state"""
        neighbour = self.dim
        for j in range(len(self.dim)):
            amplitude = (self.var_size[j][1] - self.var_size[j][0]) * 1 / 10
            delta = (-amplitude / 2.) + amplitude * np.random.random_sample()
            neighbour_point = max(min(state[j] + delta, self.var_size[j][1]), self.var_size[j][0])
            neighbour[j] = neighbour_point
        # print("neighbor", neighbour)
        return neighbour
