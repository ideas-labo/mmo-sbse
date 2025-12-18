from typing import Tuple
from typing import Dict, List, Optional
import math
import sys
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
from Code.WS.Utils.DAG_class import DAG, Task

INFRA_CONFIG = {
    "base_mips": 100,
    "vm_types": [
        {"name": "m3.medium", "factor": 3.75, "mips": 375, "bw": 85 * 1024 * 1024, "price": 0.067, "size": 10000,"ram": 4096},
        {"name": "m4.large", "factor": 7.5, "mips": 750, "bw": 35 * 1024 * 1024, "price": 0.100, "size": 10000,"ram": 4096},
        {"name": "m3.large", "factor": 7.5, "mips": 750, "bw": 85 * 1024 * 1024, "price": 0.133, "size": 10000,"ram": 4096},
        {"name": "m4.xlarge", "factor": 15.0, "mips": 1500, "bw": 65 * 1024 * 1024, "price": 0.200, "size": 10000,"ram": 4096},
        {"name": "m3.xlarge", "factor": 15.0, "mips": 1500, "bw": 125 * 1024 * 1024, "price": 0.266, "size": 10000,"ram": 4096},
        {"name": "m3.2xlarge", "factor": 30.0, "mips": 3000, "bw": 125 * 1024 * 1024, "price": 0.532, "size": 10000,"ram": 4096},
        {"name": "m4.2xlarge", "factor": 40.0, "mips": 4000, "bw": 125 * 1024 * 1024, "price": 0.400, "size": 10000,"ram": 4096},
        {"name": "m4.4xlarge", "factor": 45.0, "mips": 4500, "bw": 17.5 * 1024 * 1024, "price": 0.800, "size": 10000,"ram": 4096}
    ]
}

class MOHEFT:
    @staticmethod
    def b_rank(dag: DAG) -> Dict[int, float]:
        valid_tasks = [t for t in dag.real_tasks if t.id >= 0]
        upward_rank = {task.id: -1 for task in valid_tasks}
        max_depth = 0
        changed = True
        current_depth = 0
        while changed:
            changed = False
            for task in valid_tasks:
                if upward_rank[task.id] != -1:
                    continue
                ready = True
                for succ_id in dag.contributeTo.get(task.id, []):
                    if succ_id >= 0:
                        if upward_rank.get(succ_id, -1) == -1 or upward_rank[succ_id] == current_depth:
                            ready = False
                            break
                if ready:
                    real_succs = [s for s in dag.contributeTo.get(task.id, []) if s >= 0]
                    max_succ_rank = max([upward_rank[s] for s in real_succs], default=-1)
                    upward_rank[task.id] = max_succ_rank + 1
                    max_depth = max(max_depth, upward_rank[task.id])
                    changed = True

            current_depth += 1
        rank_dict = {}
        for depth in range(max_depth + 1):
            same_rank_tasks = [t for t in valid_tasks if upward_rank[t.id] == depth]
            same_rank_tasks.sort(key=lambda t: -len([s for s in dag.contributeTo.get(t.id, []) if s >= 0]))

            for i, task in enumerate(same_rank_tasks):
                rank_dict[task.id] = depth + (i / (len(same_rank_tasks) + 1.0))

        return rank_dict

    @staticmethod
    def generate_initial_solutions(dag: DAG, max_vms: int) -> List[Tuple]:
        solutions = []

        time_scheduler = HEFTScheduler(dag, max_vms, is_cost_optimized=False)
        time_scheduler.schedule()
        time_sol = time_scheduler.get_solution()
        if MOHEFT._validate_solution(dag, *time_sol):
            solutions.append(time_sol)

        cost_scheduler = HEFTScheduler(dag, max_vms, is_cost_optimized=True)
        cost_scheduler.schedule()
        cost_sol = cost_scheduler.get_solution()
        if MOHEFT._validate_solution(dag, *cost_sol):
            solutions.append(cost_sol)

        return solutions

    @staticmethod
    def _validate_solution(dag: DAG, task_order: List[int],
                           vm_mapping: List[int], vm_types: List[int]) -> bool:
        seen = set()
        for task_id in task_order:
            for pred in dag.requiring.get(task_id, []):
                if pred >= 0 and pred not in seen:
                    return False
            seen.add(task_id)
        if any(vm >= len(vm_types) for vm in vm_mapping):
            return False
        return True


class HEFTScheduler:
    def __init__(self, dag: DAG, max_vms: int = 10, is_cost_optimized: bool = False):
        self.dag = dag
        self.max_vms = max_vms
        self.is_cost_optimized = is_cost_optimized
        self.used_vm = 0
        self.vm_types = [0] * max_vms
        self.unit_price = [0.0] * max_vms
        self.vm_finish_time = [0.0] * max_vms
        self.task_finish_time = {}
        self.task_to_vm = {}
        self.scheduled_tasks = []
        self.makespan = -1
        self.cost = -1
        self.vm_list = [{
            'mips': cfg['mips'],
            'bw': cfg['bw'],
            'price': cfg['price'],
            'name': cfg['name']
        } for cfg in INFRA_CONFIG["vm_types"]]

    def schedule(self):
        rank = MOHEFT.b_rank(self.dag)
        ordered_tasks = sorted(
            [t for t in self.dag.real_tasks if t.id >= 0],
            key=lambda t: -rank[t.id]
        )

        for task in ordered_tasks:
            ready_time = self._calculate_ready_time(task)
            best_vm = self._select_best_vm(task, ready_time)
            if best_vm is None:
                continue
            self._assign_task(task, best_vm)

        self._schedule_exit_task()

    def _schedule_exit_task(self):
        if not self.dag.exit_task:
            return

        real_task_ids = [t.id for t in self.dag.real_tasks if t.id >= 0]
        max_finish = max((self.task_finish_time.get(tid, 0) for tid in real_task_ids), default=0)

        exit_vm = self.vm_list[0]
        exit_mips = exit_vm["mips"]
        exec_time = self.dag.exit_task.cloudlet_length / exit_mips

        self.dag.exit_task.start_time = max_finish
        self.dag.exit_task.finish_time = max_finish + exec_time

    def _calculate_ready_time(self, task: Task) -> float:
        ready_time = 0.0

        entry_transfer = self.dag.file_transfer_time.get(task.id, 0.0)
        ready_time = max(ready_time, entry_transfer)

        for pred_id in self.dag.requiring.get(task.id, []):
            if pred_id >= 0:
                pred_finish = self.task_finish_time.get(pred_id, 0.0)
                ready_time = max(ready_time, pred_finish)

        return ready_time

    def _select_best_vm(self, task: Task, ready_time: float) -> Optional[dict]:
        candidates = []
        for vm_id in range(self.used_vm):
            vm_type = self.vm_types[vm_id]
            mips = INFRA_CONFIG["vm_types"][vm_type]["mips"]
            exec_time = task.cloudlet_length / mips
            start_time = max(ready_time, self.vm_finish_time[vm_id])
            candidates.append({
                'vm_id': vm_id,
                'start_time': start_time,
                'finish_time': start_time + exec_time,
                'is_new': False
            })

        if self.used_vm < self.max_vms:
            for vm_type_idx, vm_config in enumerate(INFRA_CONFIG["vm_types"]):
                mips = vm_config["mips"]
                exec_time = task.cloudlet_length / mips
                candidates.append({
                    'vm_id': self.used_vm,
                    'start_time': ready_time,
                    'finish_time': ready_time + exec_time,
                    'is_new': True,
                    'vm_type': vm_type_idx
                })

        if not candidates:
            return None

        if self.is_cost_optimized:
            return min(candidates, key=lambda x: (
                INFRA_CONFIG["vm_types"][x.get('vm_type', self.vm_types[x['vm_id']])]["price"],
                x['finish_time']
            ))
        else:
            return min(candidates, key=lambda x: x['finish_time'])

    def _assign_task(self, task: Task, vm_info: dict):
        if vm_info['is_new']:
            self.vm_types[self.used_vm] = vm_info['vm_type']
            self.unit_price[self.used_vm] = INFRA_CONFIG["vm_types"][vm_info['vm_type']]["price"]
            self.used_vm += 1

        self.task_to_vm[task.id] = vm_info['vm_id']
        self.task_finish_time[task.id] = vm_info['finish_time']
        self.vm_finish_time[vm_info['vm_id']] = vm_info['finish_time']
        task.vm_id = vm_info['vm_id']
        task.start_time = vm_info['start_time']
        task.finish_time = vm_info['finish_time']
        self.scheduled_tasks.append(task)
        self.makespan = max(self.makespan, vm_info['finish_time'])
        self.cost = sum(
            math.ceil((self.vm_finish_time[vm_id] - 0) / 3600) * self.unit_price[vm_id]
            for vm_id in range(self.used_vm)
        )

    def get_solution(self) -> Tuple[List[int], List[int], List[int]]:
        task_order = [t.id for t in self.scheduled_tasks]
        vm_mapping = [self.task_to_vm[t.id] for t in self.scheduled_tasks]
        vm_types = self.vm_types[:self.max_vms]
        return task_order, vm_mapping, vm_types