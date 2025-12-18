import random
from jmetal.core.solution import FloatSolution
from jmetal.core.operator import Crossover, Mutation
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import DominanceComparator
import sys

from Code.SCT.Utils.get_objectives import get_objective_score_similarly, GotoFailedLabelException

sys.path.append('../')
sys.path.append('../..')

import numpy as np
selection = BinaryTournamentSelection(comparator=DominanceComparator())

class CustomProblem:
    def __init__(self, seed ,filename, mode, dict_search, independent_set, scaler_ft, scaler_fa, minimize, unique_elements_per_column):
        self.filename = filename
        self.mode = mode
        self.dict_search = dict_search
        self.independent_set = independent_set
        self.scaler_ft = scaler_ft
        self.scaler_fa = scaler_fa
        self.minimize = minimize
        self.unique_elements_per_column = unique_elements_per_column
        self.number_of_variables = len(independent_set)
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.lower_bound = [min(s) for s in independent_set]
        self.upper_bound = [max(s) for s in independent_set]
        self.seed=seed

    def evaluate(self, solution, global_pareto_front, budget1):

        decision = [solution.variables[i] for i in range(self.number_of_variables)]
        try:
            original_objectives = get_objective_score_similarly(
                decision, self.dict_search, "real",
                global_pareto_front, budget1
            )
            solution.original_objectives = original_objectives
            return solution
        except GotoFailedLabelException:
            return None

    def create_solution(self):

        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = [float(random.choice(self.independent_set[i])) for i in range(self.number_of_variables)]
        return new_solution

class UniformCrossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self, probability: float):
        super(UniformCrossover, self).__init__(probability=probability)

    def execute(self, parents: list[FloatSolution]) -> list[FloatSolution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        if not isinstance(parents[0], FloatSolution) or not isinstance(parents[1], FloatSolution):
            raise TypeError("Parents must be instances of FloatSolution")

        offspring = [FloatSolution(
            lower_bound=parents[0].lower_bound,
            upper_bound=parents[0].upper_bound,
            number_of_objectives=len(parents[0].objectives),
            number_of_constraints=len(parents[0].constraints)
        ) for _ in range(2)]

        rand = random.random()
        if rand <= self.probability:
            for i in range(len(parents[0].variables)):
                if random.random() < 0.5:
                    offspring[0].variables[i] = parents[0].variables[i]
                    offspring[1].variables[i] = parents[1].variables[i]
                else:
                    offspring[0].variables[i] = parents[1].variables[i]
                    offspring[1].variables[i] = parents[0].variables[i]
        else:
            offspring[0].variables = parents[0].variables.copy()
            offspring[1].variables = parents[1].variables.copy()

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Uniform crossover"

class BoundaryMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, independent_set):
        super(BoundaryMutation, self).__init__(probability=probability)
        self.independent_set = independent_set

    def execute(self, solution: FloatSolution) -> FloatSolution:
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                choices = self.independent_set[i]
                boundary_value = random.choice([min(choices), max(choices)])
                solution.variables[i] = boundary_value
        return solution

    def get_name(self):
        return "Boundary mutation"


class TwoPointCrossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self, probability: float):
        super(TwoPointCrossover, self).__init__(probability=probability)

    def execute(self, parents: list[FloatSolution]) -> list[FloatSolution]:
        if len(parents) != 2:
            raise Exception("The number of parents is not two: {}".format(len(parents)))

        if not isinstance(parents[0], FloatSolution) or not isinstance(parents[1], FloatSolution):
            raise TypeError("Parents must be instances of FloatSolution")

        offspring = [FloatSolution(
            lower_bound=parents[0].lower_bound,
            upper_bound=parents[0].upper_bound,
            number_of_objectives=len(parents[0].objectives),
            number_of_constraints=len(parents[0].constraints)
        ) for _ in range(2)]

        rand = random.random()
        if rand <= self.probability:
            length = len(parents[0].variables)
            point1 = random.randint(0, length - 1)
            point2 = random.randint(0, length - 1)

            if point1 > point2:
                point1, point2 = point2, point1

            for i in range(length):
                if point1 <= i <= point2:
                    offspring[0].variables[i] = parents[1].variables[i]
                    offspring[1].variables[i] = parents[0].variables[i]
                else:
                    offspring[0].variables[i] = parents[0].variables[i]
                    offspring[1].variables[i] = parents[1].variables[i]
        else:
            offspring[0].variables = parents[0].variables.copy()
            offspring[1].variables = parents[1].variables.copy()

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "Two-point crossover"