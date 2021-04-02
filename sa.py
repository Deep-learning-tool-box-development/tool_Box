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
        :param alpha: double, temperature changing step, [0.985, 0.999]
        :param var_size: list, upper and lower bounds of each parameter
        :param num_itr: int, iteration times
        """
        self.interval = (0, 1)  # set a range (0,1)
        self.objective = objective  # Objective network to be optimize
        self.initial_temp = initial_temp  # 90
        self.final_temp = final_temp  # .1
        self.alpha = alpha  # 0.92 衰减因子
        self.var_size = var_size  # [[],[],[]]
        self.dim = np.zeros(len(var_size))
        self.Global_Best = []
        self.Best_Cost = []

    def run(self):
        """
        :return: current state, current cost, state lists, cost lists
        """
        """ Optimize object network with the simulated annealing algorithm."""

        initial_state = self._random_start()  # start from a random state, multiple dimension
        current_temp = self.initial_temp
        current_state = initial_state
        cost = self.objective(current_state)
        solution = current_state
        states, costs = [current_state], [cost]
        num_itr = 1
        while current_temp > self.final_temp:
            print("iteration", num_itr, "...")
            neighbour_of_current = self._random_neighbour(solution)
            # print("now_neighbor", neighbour_of_current)
            # Check if neighbor is best so far           
            cost_current = self.objective(solution)
            cost_neighbour = self.objective(neighbour_of_current)
            cost_diff = cost_current - cost_neighbour
            print("cost_current_state", cost_current, "cost_neighbour", cost_neighbour)
            print("cost_diff", cost_diff)
            # if the new solution is better, accept it
            if cost_diff > 0:
                solution = neighbour_of_current
                cost = cost_neighbour
                # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if np.random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                    solution = neighbour_of_current
                    cost = cost_neighbour
                else: 
                  solution = current_state
                  cost = cost_current
                # decrement the temperature
            print("cost", cost)
            states.append(solution)
            costs.append(cost)
            current_temp = current_temp*self.alpha
            num_itr += 1
            print(
                "Best parameters: ",
              "\ndropout=", solution[0],
              "learning rate=", solution[1],
              "batch size=", int(solution[2]),
              "number of convolution=", solution[3])

        plt.plot(costs)
    #
    # def plot_curve(self):
    #     """
    #     Plot optimizer curve with iteration
    #     :return: None
    #     """
    #     plt.plot()

    def _random_start(self):
        """ Random point in the interval """
        print("___START____")
        rd_state = self.dim
        for i in range(len(self.dim)):
            #print("rd_point", rd_state)
            rd_point = np.random.uniform(self.var_size[i][0], self.var_size[i][1])
            rd_state[i] = rd_point
        print("init_random_state", rd_state)
        return rd_state

    def _random_neighbour(self, state):  # fraction_origin=1
        """Find neighbour of current state"""
        neighbour = self.dim
        for j in range(len(self.dim)):
            amplitude = (self.var_size[j][1] - self.var_size[j][0]) * 1 / 10
            delta = (-amplitude / 2.) + amplitude * np.random.random_sample()
            neighbour_point = max(min(state[j] + delta, self.var_size[j][1]), self.var_size[j][0])
            neighbour[j] = neighbour_point
        # print("neighbor", neighbour)
        return neighbour
