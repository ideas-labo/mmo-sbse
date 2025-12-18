from typing import List
import random
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.core.operator import Crossover, Mutation


from typing import List
import random
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.core.operator import Crossover

class ImprovedZhuCrossover(Crossover[IntegerSolution, IntegerSolution]):
    def __init__(self, probability: float, problem: IntegerProblem):
        super().__init__(probability=probability)
        self.problem = problem

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        if len(parents) != 2:
            raise ValueError("Exactly two parents are required")

        if random.random() > self.probability:
            return parents.copy()

        n = self.problem.num_valid_tasks
        parent1, parent2 = parents[0], parents[1]
        max_vm = self.problem.max_simultaneous_ins - 1

        child1 = self.problem.create_solution()
        child2 = self.problem.create_solution()

        cx_point = random.randint(1, n - 2)

        child1_order = parent2.variables[:n][:cx_point]
        for task in parent1.variables[:n]:
            if task not in child1_order:
                child1_order.append(task)

        child2_order = parent1.variables[:n][:cx_point]
        for task in parent2.variables[:n]:
            if task not in child2_order:
                child2_order.append(task)

        child1.variables[:n] = child1_order
        child2.variables[:n] = child2_order

        child1.variables[n:2 * n] = parent1.variables[n:2 * n].copy()
        child1.variables[2 * n:] = parent1.variables[2 * n:].copy()
        child2.variables[n:2 * n] = parent2.variables[n:2 * n].copy()
        child2.variables[2 * n:] = parent2.variables[2 * n:].copy()

        for i in range(cx_point):
            child1.variables[n + i] = parent2.variables[n + i]
            child2.variables[n + i] = parent1.variables[n + i]

            vm1 = min(parent2.variables[n + i], max_vm)
            vm2 = min(parent1.variables[n + i], max_vm)
            mutation_prob = 1/n

            conflict_in_child1 = any(v == vm1 for v in child1.variables[n:2 * n][cx_point:])
            if not conflict_in_child1:
                if random.random() < mutation_prob:
                    child1.variables[2 * n + vm1] = random.randint(0, 7)
            else:
                child1.variables[2 * n + vm1] = random.choice([
                    parent1.variables[2 * n + vm1],
                    parent2.variables[2 * n + vm1]
                ])

            conflict_in_child2 = any(v == vm2 for v in child2.variables[n:2 * n][cx_point:])
            if not conflict_in_child2:
                if random.random() < mutation_prob:
                    child2.variables[2 * n + vm2] = random.randint(0, 7)
            else:
                child2.variables[2 * n + vm2] = random.choice([
                    parent1.variables[2 * n + vm2],
                    parent2.variables[2 * n + vm2]
                ])

        return [child1, child2]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "ImprovedZhuCrossover"

class ImprovedZhuMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, problem: IntegerProblem):
        super().__init__(probability=probability)
        self.problem = problem

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        if random.random() > self.probability:
            return solution

        n = self.problem.num_valid_tasks
        max_vm = self.problem.max_simultaneous_ins - 1
        dag = self.problem.dag

        order = solution.variables[:n]
        pos = random.randint(0, n - 1)
        task_id = order[pos]

        start = pos
        end = pos

        while start >= 0 and order[start] not in dag.requiring.get(task_id, []):
            start -= 1

        while end < n and order[end] not in dag.contributeTo.get(task_id, []):
            end += 1

        valid_start = start + 1
        valid_end = end - 1

        if valid_start <= valid_end:
            new_pos = random.randint(valid_start, valid_end)
            if new_pos != pos:
                if new_pos < pos:
                    solution.variables[:n][new_pos+1:pos+1] = solution.variables[:n][new_pos:pos]
                else:
                    solution.variables[:n][pos:new_pos] = solution.variables[:n][pos+1:new_pos+1]
                solution.variables[:n][new_pos] = task_id

        for i in range(n):
            if random.random() < self.probability:
                solution.variables[n + i] = random.randint(0, max_vm)

        for i in range(self.problem.max_simultaneous_ins):
            if random.random() < self.probability:
                solution.variables[2*n + i] = random.randint(0, 7)

        return solution

    def get_name(self) -> str:
        return "ZhuMutation"