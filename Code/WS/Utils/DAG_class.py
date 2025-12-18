from collections import deque
from typing import Tuple
from collections import defaultdict
from typing import List, Optional
from typing import Dict
from dataclasses import dataclass


@dataclass
class Task:
    CREATED = 0
    READY = 1
    QUEUED = 2
    INEXEC = 3
    SUCCESS = 4
    FAILED = 5
    CANCELED = 6
    PAUSED = 7
    RESUMED = 8
    FAILED_RESOURCE_UNAVAILABLE = 9

    def __init__(self, id: int, cloudlet_length: int):
        self.id = id
        self.cloudlet_length = cloudlet_length
        self.earliest_start = 0.0
        self.file_transfer_end = 0.0
        self.defCloudletL = cloudlet_length
        self.status = self.CREATED
        self.exec_start_time = -1.0
        self.finish_time = -1.0
        self.cloudlet_finished_so_far = 0
        self.vm_id = -1
        self.input_files: Dict[str, int] = {}
        self.output_files: Dict[str, int] = {}
        self._earliest_start = 0.0
        self._submission_time = 0.0
        self._current_transfer_progress = 0.0
        self._last_transfer_update = 0.0
        self.file_transfer_end = 0.0
        self._transfer_progress = 0.0
        self._last_update_time = 0.0

    @property
    def submission_time(self) -> float:
        return self._submission_time

    def reset_transfer_state(self):
        self._transfer_progress = 0.0
        self._last_update_time = 0.0

    def update_transfer_progress(self, current_time: float, bw: float):
        if self.status == Task.SUCCESS:
            return

        total_size = sum(self.input_files.values())
        if self._transfer_progress >= total_size:
            self.file_transfer_end = 0.0
            return

        elapsed = current_time - self._last_update_time
        transferred = min(bw * elapsed, total_size - self._transfer_progress)
        self._transfer_progress += transferred
        self._last_update_time = current_time

        remaining = total_size - self._transfer_progress
        self.file_transfer_end = current_time + (remaining / bw) if bw > 0 else current_time

    @submission_time.setter
    def submission_time(self, value: float):
        if value >= 0:
            self._submission_time = value

    @property
    def earliest_start(self):
        return self._earliest_start

    @earliest_start.setter
    def earliest_start(self, value):
        self._earliest_start = value

    def get_remaining_cloudlet_length(self) -> int:
        return max(self.cloudlet_length - self.cloudlet_finished_so_far, 0)

    def set_cloudlet_length(self, length: int) -> bool:
        if length <= 0:
            return False
        self.cloudlet_length = length
        return True

    def set_cloudlet_finished_so_far(self, length: int):
        if length >= 0:
            self.cloudlet_finished_so_far = length

    def set_status(self, new_status: int, current_time: float = -1):
        if self.status == new_status:
            return
        if new_status == self.SUCCESS and current_time >= 0:
            self.finish_time = current_time
        self.status = new_status

    def set_exec_start_time(self, time: float):
        self.exec_start_time = time

    def set_finish_time(self, time: float):
        self.finish_time = time

    def setExecStartTime(self, time: float):
        self.set_exec_start_time(time)

    @property
    def runtime(self) -> float:
        return self.cloudlet_length / 1000.0


class DAG:
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.real_tasks: List[Task] = []
        self.requiring: Dict[int, List[int]] = defaultdict(list)
        self.contributeTo: Dict[int, List[int]] = defaultdict(list)
        self.file_transfers: Dict[Tuple[int, int], int] = {}
        self.file_transfer_time: Dict[int, float] = {}
        self.entry_id = -1
        self.exit_id = -2
        self.entry_task: Optional[Task] = None
        self.exit_task: Optional[Task] = None
        self.ready: List[int] = []
        self.totalNeed: List[int] = []
        self.totalCloudletNum: int = 0
        self.defultTotalCloudletNum: int = 0
        self.debug = False

    def add_dependency(self, from_task: Task, to_task: Task, file_size: int = 0):
        if from_task.id not in self.tasks:
            self.tasks[from_task.id] = from_task
            if from_task.id >= 0:
                self._add_real_task(from_task)

        if to_task.id not in self.tasks:
            self.tasks[to_task.id] = to_task
            if to_task.id >= 0:
                self._add_real_task(to_task)

        if to_task.id not in self.requiring:
            self.requiring[to_task.id] = []
        if from_task.id not in self.requiring[to_task.id]:
            self.requiring[to_task.id].append(from_task.id)

        if from_task.id not in self.contributeTo:
            self.contributeTo[from_task.id] = []
        if to_task.id not in self.contributeTo[from_task.id]:
            self.contributeTo[from_task.id].append(to_task.id)

        if file_size > 0 and to_task.id != self.exit_id:
            self.file_transfers[(from_task.id, to_task.id)] = file_size

        if to_task.id >= 0:
            idx = next((i for i, t in enumerate(self.real_tasks) if t.id == to_task.id), -1)
            if idx >= 0:
                self.ready[idx] += 1
                self.totalNeed[idx] += 1

    def _add_real_task(self, task: Task, job_id: str = None):
        if task.id < 0:
            raise ValueError("Real task ID must be >= 0")
        self.tasks[task.id] = task
        self.real_tasks.append(task)
        self.ready.append(0)
        self.totalNeed.append(0)
        self.totalCloudletNum += 1
        self.defultTotalCloudletNum += 1
        if job_id is not None:
            self.job_id_map[job_id] = task.id

    def validate_workflow(self):
        entry_connections = len(self.contributeTo[self.entry_id])
        exit_connections = len(self.requiring[self.exit_id])
        if self.debug:
            print("\n=== DAG integrity check ===")
            print(f"Entry connections: {entry_connections}")
            print(f"Exit connections: {exit_connections}")
        error_count = 0
        for task in self.real_tasks:
            preds = self.requiring.get(task.id, [])
            if not any(p >= 0 for p in preds) and self.entry_id not in preds:
                if self.debug:
                    print(f"Error Task {task.id}: no valid predecessors and not connected to entry")
                error_count += 1
            succs = self.contributeTo.get(task.id, [])
            if not any(s >= 0 for s in succs) and self.exit_id not in succs:
                if self.debug:
                    print(f"Error Task {task.id}: no valid successors and not connected to exit")
                error_count += 1
        if error_count == 0:
            if self.debug:
                print("All real tasks are correctly connected to entry/exit")
        else:
            raise RuntimeError(f"Found {error_count} connection errors")

    def after_one_cloudlet_success(self, task_id: int):
        real_task_ids = [t.id for t in self.real_tasks]

        for succ_id in self.contributeTo.get(task_id, []):
            if succ_id < 0:
                continue

            try:
                idx = real_task_ids.index(succ_id)
                self.ready[idx] -= 1

                if self.ready[idx] == 0:
                    self.tasks[succ_id].set_status(Task.READY)
                    if self.debug:
                        print(f"  Task {succ_id} dependencies satisfied, set to READY")
            except ValueError:
                print(f"!! Error: index for task {succ_id} not found")

    def calc_file_transfer_times(self, task2ins: List[int], vmlist: List[dict]) -> None:
        if self.debug:
            print("\n=== Calculating file transfer times ===")
            print(f"task2ins length: {len(task2ins)}")
            print(f"vmlist length: {len(vmlist)}")
            print(f"real tasks count: {len(self.real_tasks)}")
            print(f"task id range: {min(t.id for t in self.real_tasks)}-{max(t.id for t in self.real_tasks)}")

        self.file_transfer_time = defaultdict(float)

        if self.debug:
            print("\nFile transfer relations:")
        for (src, dst), size in self.file_transfers.items():
            if self.debug:
                print(f"  {src} -> {dst} (size: {size} bytes)")

        if self.debug:
            print("\nTask predecessor relations:")
        for task in self.real_tasks:
            preds = self.requiring.get(task.id, [])
            if self.debug:
                print(f"  Task {task.id} predecessors: {preds}")

        for dst_task in (t for t in self.real_tasks if t.id != self.exit_id):
            if self.debug:
                print(f"\nProcessing destination task {dst_task.id}:")

            if (self.entry_id, dst_task.id) in self.file_transfers:
                if self.debug:
                    print(f"  Found transfer from entry to {dst_task.id}, skipping calculation")
                self.file_transfer_time[dst_task.id] = 0.0
                continue

            for src_id in (p for p in self.requiring.get(dst_task.id, []) if p >= 0):
                if self.debug:
                    print(f"  Checking predecessor {src_id} -> {dst_task.id}")

                if (src_id, dst_task.id) in self.file_transfers:
                    if self.debug:
                        print(f"    Found file transfer {src_id} -> {dst_task.id}")

                    try:
                        src_vm = task2ins[src_id]
                        dst_vm = task2ins[dst_task.id]
                        if self.debug:
                            print(f"    src_vm={src_vm}, dst_vm={dst_vm}")

                        if src_vm != dst_vm:
                            file_size = self.file_transfers[(src_id, dst_task.id)]
                            if self.debug:
                                print(f"    file size: {file_size} bytes")

                            if src_vm >= len(vmlist) or dst_vm >= len(vmlist):
                                if self.debug:
                                    print(f"    Error: VM index out of range (src_vm={src_vm}, dst_vm={dst_vm})")
                                continue

                            bw = min(vmlist[src_vm]['bw'], vmlist[dst_vm]['bw'])
                            transfer_time = file_size / bw
                            if self.debug:
                                print(f"    bandwidth: {bw} bytes/s, transfer time: {transfer_time:.2f}s")

                            self.file_transfer_time[dst_task.id] += transfer_time
                        else:
                            if self.debug:
                                print("    Same VM, skipping transfer calculation")
                    except IndexError as e:
                        print(f"    Index error: {e}")
                        print(f"    task2ins length: {len(task2ins)}, attempted access src_id={src_id} or dst_id={dst_task.id}")
                        raise
                else:
                    if self.debug:
                        print(f"    No file transfer {src_id} -> {dst_task.id}")

        if self.debug:
            print("\nFinal file transfer times:")
        for task_id, time in self.file_transfer_time.items():
            if self.debug:
                print(f"  Task {task_id}: {time:.2f}s")

    def is_task_ready(self, task: Task, current_time: float) -> bool:
        idx = next((i for i, t in enumerate(self.real_tasks) if t.id == task.id), -1)
        if idx == -1 or self.ready[idx] != 0:
            return False

        return True

    def set_cloudlet_num(self, num: int):
        self.totalCloudletNum = num
        self.defultTotalCloudletNum = num

    def validate_dependencies(self) -> bool:
        error_count = 0
        entry_conn = len(self.contributeTo.get(self.entry_id, []))
        exit_conn = len(self.requiring.get(self.exit_id, []))
        if self.debug:
            print("\n=== Dependency validation ===")
            print(f"Entry connections: {entry_conn}")
            print(f"Exit connections: {exit_conn}")
        for task in self.real_tasks:
            preds = self.requiring.get(task.id, [])
            succs = self.contributeTo.get(task.id, [])
            valid_preds = [p for p in preds if p in self.tasks]
            valid_succs = [s for s in succs if s in self.tasks]
            if not valid_preds and task.id != self.entry_id:
                if self.debug:
                    print(f"Error: Task {task.id} has no valid predecessors")
                error_count += 1
            if not valid_succs and task.id != self.exit_id:
                if self.debug:
                    print(f"Error: Task {task.id} has no valid successors")
                error_count += 1
        for (src, dst), size in self.file_transfers.items():
            if src not in self.tasks or dst not in self.tasks:
                if self.debug:
                    print(f"Error: File transfer {src}->{dst} references non-existent task")
                error_count += 1
        if error_count == 0:
            if self.debug:
                print("All dependency checks passed")
        else:
            if self.debug:
                print(f"Found {error_count} dependency errors")
        return error_count == 0

    def add_virtual_task(self, task: Task, name: str):
        if task.id >= 0:
            raise ValueError("Virtual task ID must be negative")

        self.tasks[task.id] = task
        if name == "entry":
            self.entry_id = task.id
            self.entry_task = task
            task.set_cloudlet_length(10000)
            task.vm_id = 0
            self.ready.append(0)
            self.totalNeed.append(0)
        elif name == "exit":
            self.exit_id = task.id
            self.exit_task = task
            task.set_cloudlet_length(1000)
            self.ready.append(0)
            self.totalNeed.append(0)
        else:
            raise ValueError("Virtual task name must be 'entry' or 'exit'")

    def rm_cache(self):
        self.totalCloudletNum = self.defultTotalCloudletNum
        self.ready = self.totalNeed.copy()

    def get_topological_order(self) -> List[int]:
        in_degree = {t.id: 0 for t in self.real_tasks}
        adj_list = defaultdict(list)
        for task in self.real_tasks:
            for succ in self.contributeTo.get(task.id, []):
                if succ >= 0:
                    adj_list[task.id].append(succ)
                    in_degree[succ] += 1
        queue = deque([t.id for t in self.real_tasks if in_degree[t.id] == 0])
        topo_order = []
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(topo_order) != len(self.real_tasks):
            raise RuntimeError("Workflow contains cyclic dependencies")
        return topo_order

    def _is_path_exists(self, start: int, end: int) -> bool:
        visited = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == end:
                return True
            if current not in visited:
                visited.add(current)
                for neighbor in self.contributeTo.get(current, []):
                    stack.append(neighbor)
        return False