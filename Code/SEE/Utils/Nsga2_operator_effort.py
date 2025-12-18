import numpy as np
from jmetal.core.operator import Selection, Crossover, Mutation
import random
import copy

class SEEDoubleSimpleRandomMutation(Mutation):
    def __init__(self, probability_per_variable, random_generator=None):
        self.mutation_probability = probability_per_variable
        self.random_generator = random_generator if random_generator else random.random

    def execute(self, solution):
        num_vars = len(solution.variables)
        for i in range(num_vars):
            if self.random_generator() <= self.mutation_probability:
                lb = solution.lower_bound[i]
                ub = solution.upper_bound[i]
                solution.variables[i] = lb + (ub - lb) * self.random_generator()
        return solution

    def get_name(self):
        return "SEEDoubleSimpleRandomMutation"

class SEESelection(Selection):
    def execute(self, population, k=None):
        if k is None:
            k = len(population)
        valid_pop = [s for s in population if not any(np.isinf(obj) for obj in s.objectives)]

        if not valid_pop:
            return population[:k]

        objectives = np.array([s.objectives for s in valid_pop])
        max_obj = np.max(objectives, axis=0)
        min_obj = np.min(objectives, axis=0)
        max_obj[max_obj == 0] = 1

        norm_obj = (objectives - min_obj) / (max_obj - min_obj + 1e-8)
        fitness = np.sum(norm_obj, axis=1)

        total = fitness.sum()
        selected = []
        for _ in range(k):
            r = random.uniform(0, total)
            cumulative = 0
            for i, f in enumerate(fitness):
                cumulative += f
                if cumulative >= r:
                    selected.append(valid_pop[i])
                    break
        return selected
    def get_name(self):
        return "SEESelection"


class SEESinglePointCrossover(Crossover):
    def __init__(self, crossover_probability, random_generator=None, point_generator=None):
        super().__init__(probability=crossover_probability)
        self.crossover_probability = crossover_probability
        self.random_generator = random_generator if random_generator else random.random
        self.point_generator = point_generator if point_generator else lambda a, b: random.randint(a, b)

    def execute(self, parents):
        if parents is None:
            raise ValueError("Null parameter")
        if len(parents) != 2:
            raise ValueError("There must be two parents instead of " + str(len(parents)))

        father = parents[0] if not isinstance(parents[0], list) else parents[0][0]
        mother = parents[1] if not isinstance(parents[1], list) else parents[1][0]

        offspring1 = copy.deepcopy(father)
        offspring2 = copy.deepcopy(mother)

        if self.random_generator() < self.crossover_probability:
            num_vars = len(father.variables)
            crossover_point = self.point_generator(0, num_vars - 1)

            for i in range(crossover_point):
                offspring1.variables[i] = mother.variables[i]
                offspring2.variables[i] = father.variables[i]

            for i in range(crossover_point, num_vars):
                offspring1.variables[i] = father.variables[i]
                offspring2.variables[i] = mother.variables[i]

        return [offspring1, offspring2]

    def get_name(self):
        return "SEESinglePointCrossover"

    def get_number_of_parents(self):
        return 2

    def get_number_of_children(self):
        return 2


class BoundaryMutation(Mutation):
    def __init__(self, probability, problem):
        super().__init__(probability)
        self.problem = problem

    def execute(self, solution):
        for i in range(len(solution.variables)):
            if random.random() < self.probability:
                if random.random() < 0.5:
                    solution.variables[i] = self.problem.lower_bound[i]
                else:
                    solution.variables[i] = self.problem.upper_bound[i]
        return solution

    def get_name(self):
        return "Boundary mutation"


class UniformCrossover(Crossover):
    def __init__(self, probability):
        super().__init__(probability)
        self.number_of_parents = 2
        self.number_of_children = 2

    def execute(self, parents):
        father = parents[0]
        mother = parents[1]
        if isinstance(father, list):
            father = father[0]
        if isinstance(mother, list):
            mother = mother[0]

        child1 = copy.deepcopy(father)
        child2 = copy.deepcopy(mother)

        if random.random() < self.probability:
            for i in range(len(child1.variables)):
                if random.random() < 0.5:
                    child1.variables[i], child2.variables[i] = child2.variables[i], child1.variables[i]
        return [child1, child2]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Uniform crossover"