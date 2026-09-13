from random import *
from time import time
import numpy as np
from random import uniform
import sys

class formulated_IKP:
    def __init__(self, R) -> None:
        self.R = R  # Radius of Plates
        self.pi = np.pi
        self.deg_to_rad = self.pi / 180

    def IKP(self, x):
        p = x[:3]
        theta = x[3:]

        # Attach Points on base and end-effector
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))

        a[0] = self.R[0] * np.array([0, 1, 0])
        a[1] = self.R[0] * np.array(
            [np.cos(210 * self.deg_to_rad), np.sin(210 * self.deg_to_rad), 0]
        )
        a[2] = self.R[0] * np.array(
            [np.cos(330 * self.deg_to_rad), np.sin(330 * self.deg_to_rad), 0]
        )

        b[0] = self.R[1] * np.array([0, -1, 0])
        b[1] = self.R[1] * np.array(
            [np.cos(30 * self.deg_to_rad), np.sin(30 * self.deg_to_rad), 0]
        )
        b[2] = self.R[1] * np.array(
            [np.cos(150 * self.deg_to_rad), np.sin(150 * self.deg_to_rad), 0]
        )

        # Rotation Matrix
        Q = np.array(
            [
                [
                    np.cos(theta[2]) * np.cos(theta[1]),
                    np.cos(theta[2]) * np.sin(theta[1]) * np.sin(theta[0])
                    - np.sin(theta[2]) * np.cos(theta[0]),
                    np.cos(theta[2]) * np.sin(theta[1]) * np.cos(theta[0])
                    + np.sin(theta[2]) * np.sin(theta[0]),
                ],
                [
                    np.sin(theta[2]) * np.cos(theta[1]),
                    np.sin(theta[2]) * np.sin(theta[1]) * np.sin(theta[0])
                    + np.cos(theta[2]) * np.cos(theta[0]),
                    np.sin(theta[2]) * np.sin(theta[1]) * np.cos(theta[0])
                    - np.cos(theta[2]) * np.sin(theta[0]),
                ],
                [
                    -np.sin(theta[1]),
                    np.cos(theta[1]) * np.sin(theta[0]),
                    np.cos(theta[1]) * np.cos(theta[0]),
                ],
            ]
        )

        self.l = np.zeros(6)

        self.l[0] = np.sqrt(np.dot(p + Q @ b[1] - a[0], p + Q @ b[1] - a[0]))
        self.l[1] = np.sqrt(np.dot(p + Q @ b[2] - a[0], p + Q @ b[2] - a[0]))
        self.l[2] = np.sqrt(np.dot(p + Q @ b[0] - a[1], p + Q @ b[0] - a[1]))
        self.l[3] = np.sqrt(np.dot(p + Q @ b[2] - a[1], p + Q @ b[2] - a[1]))
        self.l[4] = np.sqrt(np.dot(p + Q @ b[0] - a[2], p + Q @ b[0] - a[2]))
        self.l[5] = np.sqrt(np.dot(p + Q @ b[1] - a[2], p + Q @ b[1] - a[2]))

        return self.l

class Genetic:
    def __init__(self, radii, lowLim, highLim, numPopulation, crossoverRate, mutationRate, numIteration, point) -> None:
        self.step = 0.05
        self.radii = radii
        self.order = 6
        self.domain = np.vstack((lowLim, highLim)).T
        self.N_population = numPopulation
        self.P_cross = crossoverRate
        self.P_mut = mutationRate
        self.N_iteration = numIteration
        self.point = point
        self.initial_population()

    def initial_population(self):
        self.population = []
        for _ in range(self.N_population):
            # Chromosome = list(uniform(*self.domain[i].tolist())
            #                   for i in range(self.order))
            Chromosome = [
                np.random.choice(np.arange(self.domain[i][0], self.domain[i][1] + self.step, self.step))
                for i in range(self.order)
            ]
            self.population.append([Chromosome, 0, None])


    def crossover(self):
        self.crossed = []
        for i in range(1, self.N_population, 2):
            parents = [self.population[i-1], self.population[i]]
            if random() <= self.P_cross:
                slicePoint = randint(0, self.order - 1)
                child1 = parents[0][0][:slicePoint] + \
                    parents[1][0][slicePoint:]
                child2 = parents[1][0][:slicePoint] + \
                    parents[0][0][slicePoint:]
                self.crossed.extend([[child1, 0, None], [child2, 0, None]])

            else:
                self.crossed.extend(parents)

        if self.N_population % 2:
            self.crossed.append(self.population[-1])

        self.population.clear()
        self.population = self.crossed.copy()
        

    def mutation(self):
        for i in range(self.N_population):
            if random() <= self.P_mut:
                slicePoint = randint(0, self.order - 1)
                substitution = np.random.choice(np.arange(self.domain[slicePoint][0], self.domain[slicePoint][1] + self.step, self.step))
                self.population[i][0][slicePoint] = substitution

    def rotation_matrix(self, theta: np.array) -> np.array:     # input (theta, phi, psi)
        Q = np.array([                                          # standard Euler Z-Y-X (yaw-pitch-roll) (psi, phi, theta)
            [np.cos(theta[2]) * np.cos(theta[1]), np.cos(theta[2]) * np.sin(theta[1]) * np.sin(theta[0]) - np.sin(theta[2]) * np.cos(theta[0]), np.cos(theta[2]) * np.sin(theta[1]) * np.cos(theta[0]) + np.sin(theta[2]) * np.sin(theta[0])],
            [np.sin(theta[2]) * np.cos(theta[1]), np.sin(theta[2]) * np.sin(theta[1]) * np.sin(theta[0]) + np.cos(theta[2]) * np.cos(theta[0]), np.sin(theta[2]) * np.sin(theta[1]) * np.cos(theta[0]) - np.cos(theta[2]) * np.sin(theta[0])],
            [-np.sin(theta[1]), np.cos(theta[1]) * np.sin(theta[0]), np.cos(theta[1]) * np.cos(theta[0])]
        ])
        return Q
    
    def transformation_matrix(self, pose: list):
        pose = np.array(pose)
        P = pose[:3].reshape(3, 1)
        th = pose[3:]
        R = self.rotation_matrix(th)
        T = np.block([[R, P], [np.zeros((1, 3)), 1]])
        return T
    
    def reverse_rotation(self, R: np.array) -> list:
        beta = np.arcsin(-R[2, 0])
        gamma = np.arctan2(R[2, 1]/np.cos(beta), R[2, 2]/np.cos(beta))
        alpha = np.arctan2(R[1, 0]/np.cos(beta), R[0, 0]/np.cos(beta))
        return [gamma, beta, alpha]     # x, y, z
    
    def global2local(self, global_point, mid_plate):
        T = self.transformation_matrix(global_point)
        T1 = self.transformation_matrix(mid_plate)
        T2 = np.linalg.inv(T1) @ T
        local_point = np.hstack((T2[:3, 3], self.reverse_rotation(T2[:3, :3])))
        return local_point

    def fitness(self):
        for i in range(self.N_population):
            pose = self.population[i][0]
            test = formulated_IKP(self.radii[:2])
            links_1 = test.IKP(pose)

            test = formulated_IKP(self.radii[1:])
            end_point_local = self.global2local(self.point, pose) 
            links_2 = test.IKP(end_point_local)

            w = 1.1
            links = np.append(w*links_1, links_2)
            overall_length = np.sum(links)
            fit = 1/(1+overall_length)
            self.population[i][1] = fit
            self.population[i][2] = [links_1, links_2]

    def sorter(self):
        self.population = sorted(
            self.population, key=lambda x: x[1], reverse=True)
        
    def display_progress_bar(self, iteration, bar_length=50):
        progress = iteration / self.N_iteration
        block = int(bar_length * progress)
        bar = "#" * block + "-" * (bar_length - block)
        sys.stdout.write(f"\rIteration {iteration}/{self.N_iteration} [{bar}]")
        sys.stdout.flush()

    def main(self):
        for i in range(self.N_iteration):
            self.crossover()
            self.mutation()
            self.fitness()
            self.sorter()
            # self.display_progress_bar(i+1)
            print(f'Iteration {i:03}/{self.N_iteration}: mid-plate-pose:', np.round(self.population[0][0], 4))

        print('\nmid-plate-pose:', np.round(self.population[0][0], 4))
        print('first sg links:\t', np.round(self.population[0][2][0], 4))
        print('second sg links:', np.round(self.population[0][2][1], 4))
        print('score:', round(self.population[0][1], 4))
        print('worst score:', round(self.population[-1][1], 4))
        return np.hstack((self.population[0][0], self.population[0][2][0], self.population[0][2][1]))




# test case 1 ----------------------------------------------------------------------------------------
start = time()
point = [0, 0, 3, 0, 0, 0]
lowLim = [-1, -1, 0.5, -np.pi/6, -np.pi/6, -np.pi/4]
highLim = [1, 1, 2, np.pi/6, np.pi/6, np.pi/4]
radii = [0.5, 0.5, 0.5]
sample = Genetic(radii, lowLim, highLim, 1000, 0.8, 0.8, 500, point)
for i in range(sample.N_population):
    print(np.round(sample.population[i][0], 4))
print()
sample.main()
stop = time()
print(f'runtime = {round(stop - start, 4)} sec')
