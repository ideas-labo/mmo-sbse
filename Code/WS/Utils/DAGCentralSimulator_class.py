import heapq
from typing import Dict, List, Optional
import sys
sys.path.append('../')
sys.path.append('../..')
from Code.WS.Utils.DAG_class import DAG, Task


class DAGCentralSimulator:
    def __init__(self):
        self.dag: Optional[DAG] = None
        self.vm_list: List[Dict] = []
        self.vm_infos: Dict[int, Dict] = {}
        self.current_time = 0.0
        self.finished_tasks: List[Task] = []
        self.submitted_tasks = 0
        self.cycle = 0
        self.debug = False


    def set_vm_list(self, vmlist: List[Dict]):
        if not vmlist:
            raise ValueError("VM列表不能为空")
        vm_ids = [vm['id'] for vm in vmlist]
        if len(vm_ids) != len(set(vm_ids)):
            raise ValueError(f"存在重复的VM ID: {vm_ids}")
        self.vm_list = vmlist
        self.vm_infos = {
            vm['id']: {'executing': None, 'waiting_list': [], 'mips': vm['mips'], 'bw': vm['bw'], 'last_available': 0.0
                       } for vm in vmlist}
        self.max_vms = max([vm['id'] for vm in self.vm_list] or [0]) + 1
        if self.debug:
            print("\n=== VM Initialization Details ===")
            print(f"Total VMs: {len(vmlist)}")
            print("ID\tMIPS\tBandwidth")
            for vm in sorted(vmlist, key=lambda x: x['id']):
                print(f"{vm['id']}\t{vm['mips']}\t{vm['bw']}")
            print(f"Max VM ID: {max(vm_ids)}\n")

    def set_cloudlet_dag(self, dag: DAG):
        if self.debug:
            print(f"[Init] Set DAG (Total Tasks: {dag.totalCloudletNum})")
        self.dag = dag
        self.dag.rm_cache()
        for task in self.dag.real_tasks:
            task.set_cloudlet_finished_so_far(0)
            task.set_status(Task.CREATED)
            task.set_exec_start_time(-1.0)
            task.set_finish_time(-1.0)
        if self.debug:
            print("Task Dependency Verification:")
            for task in self.dag.real_tasks[:5]:
                preds = self.dag.requiring.get(task.id, [])
                print(f"  Task {task.id} Depends on: {preds}")

    def _calculate_makespan(self) -> float:
        real_finished = [t for t in self.finished_tasks if t.id >= 0]
        if not real_finished:
            return 0.0
        return max(t.finish_time for t in real_finished)

    def task_submit(self, task: Task, file_transfer_time: float) -> bool:
        if self.debug:
            print(
                f"[DEBUG] Submit Task {task.id} → VM {task.vm_id} (Length: {task.cloudlet_length} MI, Transfer Time: {file_transfer_time:.2f}s)")
            if task.vm_id not in self.vm_infos:
                print(f"!! Error: Task {task.id} Assigned to Invalid VM {task.vm_id}")
                return False
        self.submitted_tasks += 1

        if task.id == self.dag.entry_id:
            assigned_vm = 0
            vm_mips = self.vm_infos[assigned_vm]['mips']
            exec_time = task.cloudlet_length / vm_mips

            task.set_exec_start_time(self.current_time)
            task.set_finish_time(self.current_time + exec_time)
            self.vm_infos[assigned_vm]['executing'] = task
            task.set_status(Task.INEXEC)
            return True

        elif task.id == self.dag.exit_id:
            return True

        task.submission_time = self.current_time

        if file_transfer_time > 0:
            extra_length = int(file_transfer_time * self.vm_infos[task.vm_id]['mips'])
            task.set_cloudlet_length(task.defCloudletL + extra_length)
            if self.debug:
                print(f"  Task {task.id} Adjust Length: {task.defCloudletL} -> {task.cloudlet_length}"
                      f"(Transfer Time: {file_transfer_time:.2f}s)")

        task.file_transfer_end = self.current_time + file_transfer_time
        task.earliest_start = task.file_transfer_end

        vmlist = self.vm_infos[task.vm_id]['waiting_list']
        vmlist.append(task)
        if self.debug:
            print(f"[Confirm] Task {task.id} Added to VM {task.vm_id} Queue Tail")
        task.set_status(Task.QUEUED)
        return True

    def _print_status_report(self):
        print(f"\n=== Status Report @ {self.current_time:.2f}s ===")
        print(f"Total Progress: {len(self.finished_tasks)}/{self.dag.totalCloudletNum}")

        for vm_id, vm in sorted(self.vm_infos.items()):
            status = "Idle" if vm['executing'] is None else \
                f"Executing Task {vm['executing'].id} (Remaining: {vm['executing'].get_remaining_cloudlet_length()} MI)"
            print(f"  VM {vm_id}: {status} | Waiting Queue: {len(vm['waiting_list'])}")
        unfinished = [t for t in self.dag.real_tasks if t.id not in [f.id for f in self.finished_tasks]]
        if unfinished:
            print("\nBlocked Task Analysis (Top 5):")
            for task in unfinished[:5]:
                blocking = []
                for pred in self.dag.requiring.get(task.id, []):
                    if pred not in [t.id for t in self.finished_tasks]:
                        blocking.append(pred)
                print(f"  Task {task.id} Waiting for: {blocking} or Unfinished File Transfer")



    def boot(self) -> bool:
        if not self.dag:
            print("!! Error: DAG Not Set")
            return False

        self.current_time = 0.0
        self.finished_tasks = []
        self.cycle = 0
        self.submitted_tasks = 0

        if self.debug:
            print(f"\n=== Start DAG Scheduling (Total Tasks: {self.dag.totalCloudletNum}) ===")
            print(f"VM Count: {len(self.vm_list)}")
            print(f"Initial Time: {self.current_time:.2f}s")

        entry_task = self.dag.entry_task
        self.task_submit(entry_task, 0.0)
        vm_mips = self.vm_infos[entry_task.vm_id]['mips']
        exec_time = entry_task.cloudlet_length / vm_mips
        event_queue = []
        heapq.heappush(event_queue, (exec_time, 'TASK_FINISH', entry_task))

        while len(self.finished_tasks) < self.dag.totalCloudletNum + 2:
            if not event_queue:
                self.current_time += 0.1
                heapq.heappush(event_queue, (self.current_time, 'SCHEDULE', None))
                continue

            event_time, event_type, event_data = heapq.heappop(event_queue)
            self.current_time = event_time
            self.cycle += 1

            if event_type == 'TASK_FINISH':
                self._process_finish_event(event_queue, event_data)
            elif event_type == 'SCHEDULE':
                self._process_schedule_event(event_queue)

            if self.cycle % 10 == 0 and self.debug:
                self._print_status_report()

        real_finished = [t for t in self.finished_tasks if t.id >= 0]
        success = len(real_finished) == self.dag.totalCloudletNum

        if self.debug:
            print("\n=== Scheduling Result ===")
            print(f"Total Time: {self.current_time:.2f}s")
            print(f"Total Cycles: {self.cycle}")
            print(f"Completed Tasks: {len(real_finished)}/{self.dag.totalCloudletNum}")
            print("Status:", "Success" if success else "Failed")

        return success

    def _process_finish_event(self, event_queue, finished_task):
        finished_task.set_status(Task.SUCCESS, self.current_time)
        finished_task.set_cloudlet_finished_so_far(finished_task.cloudlet_length)
        self.finished_tasks.append(finished_task)

        if self.debug:
            print(f"\n[CYCLE {self.cycle}] Task {finished_task.id} Completed @ {self.current_time:.2f}s")
            print(f"  Release VM {finished_task.vm_id} Resources")

        self.vm_infos[finished_task.vm_id]['executing'] = None

        self.dag.after_one_cloudlet_success(finished_task.id)

        if self.debug:
            preds = self.dag.requiring.get(finished_task.id, [])
            succs = self.dag.contributeTo.get(finished_task.id, [])
            print(f" Task {finished_task.id} Update Dependencies: Predecessors={preds}, Successors={succs}")

        real_finished = [t.id for t in self.finished_tasks if t.id >= 0]
        if len(real_finished) == self.dag.totalCloudletNum and \
                self.dag.exit_task.id not in [t.id for t in self.finished_tasks]:
            self._schedule_exit_task(event_queue)
            return

        heapq.heappush(event_queue, (self.current_time + 0.01, 'SCHEDULE', None))

        if self.debug:
            print("  Added Scheduling Event @", self.current_time + 0.01)

    def _process_schedule_event(self, event_queue):
        for vm_id, vm_info in self.vm_infos.items():
            if vm_info['executing'] is not None:
                continue

            for i in range(len(vm_info['waiting_list']) - 1, -1, -1):
                task = vm_info['waiting_list'][i]
                if self.dag.is_task_ready(task, self.current_time):
                    del vm_info['waiting_list'][i]
                    self._start_task_execution(vm_id, task, event_queue)
                    break

    def _start_task_execution(self, vm_id: int, task: Task, event_queue):
        vm_info = self.vm_infos[vm_id]
        vm_info['executing'] = task

        start_time = max(
            self.current_time,
            task.file_transfer_end,
            task.earliest_start
        )
        exec_time = task.cloudlet_length / vm_info['mips']
        finish_time = start_time + exec_time

        task.set_exec_start_time(start_time)
        task.set_finish_time(finish_time)
        task.set_status(Task.INEXEC)

        heapq.heappush(event_queue, (finish_time, 'TASK_FINISH', task))

        if self.debug:
            print(f"  VM {vm_id} ← Task {task.id} [Activated]")
            print(f"    Start: {start_time:.2f}s, Duration: {exec_time:.2f}s")

    def _schedule_exit_task(self, event_queue):
        if not self.dag.exit_task or self.dag.exit_task.id in [t.id for t in self.finished_tasks]:
            return

        real_finished = [t.id for t in self.finished_tasks if t.id >= 0]
        if len(real_finished) != self.dag.totalCloudletNum:
            return

        exit_task = self.dag.exit_task
        exit_task.vm_id = 0
        vm_mips = self.vm_infos[0]['mips']
        exec_time = exit_task.cloudlet_length / vm_mips
        finish_time = self.current_time + exec_time

        exit_task.set_exec_start_time(self.current_time)
        exit_task.set_finish_time(finish_time)
        self.vm_infos[0]['executing'] = exit_task

        heapq.heappush(event_queue, (finish_time, 'TASK_FINISH', exit_task))

        if self.debug:
            print(f"[EXIT] Schedule Exit Task @ {self.current_time:.2f}s, Expected Completion: {finish_time:.2f}s")

    def _get_real_finished_tasks(self):
        return [t for t in self.finished_tasks if t.id >= 0]