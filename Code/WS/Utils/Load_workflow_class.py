from typing import Tuple
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List
import sys
sys.path.append('../')
sys.path.append('../..')
sys.path.append('../../..')
from Code.WS.Utils.DAG_class import Task, DAG


class PegasusWorkflowLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ns = {'pegasus': 'http://pegasus.isi.edu/schema/DAX'}
        self._file_producers: Dict[str, int] = {}
        self._file_sizes: Dict[str, int] = {}
        self._task_map: Dict[str, Task] = {}
        self.job_id_counter = 0

    def load(self) -> Tuple[List[Task], DAG]:
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            dag = DAG()
            self.job_id_counter = 0
            entry_task = Task(-1, 10000)
            exit_task = Task(-2, 1000)
            dag.add_virtual_task(entry_task, "entry")
            dag.add_virtual_task(exit_task, "exit")
            self._task_map["entry"] = entry_task
            self._task_map["exit"] = exit_task
            for job in root.findall('.//pegasus:job', self.ns):
                self._create_real_task(job, dag)
            for child in root.findall('.//pegasus:child', self.ns):
                self._process_dependencies(child, dag)
            self._link_orphan_tasks(dag)
            dag.set_cloudlet_num(len(dag.real_tasks))
            return list(dag.tasks.values()), dag
        except Exception as e:
            raise RuntimeError(f"Workflow loading failed: {str(e)}")

    def _create_real_task(self, job: ET.Element, dag: DAG):
        runtime = float(job.get('runtime', '0'))
        job_id = job.get('id')
        task = Task(self.job_id_counter, int(runtime * 1000))
        self.job_id_counter += 1
        for uses in job.findall('.//pegasus:uses', self.ns):
            self._process_file_usage(uses, task)
        dag._add_real_task(task)
        self._task_map[job_id] = task

    def _process_file_usage(self, uses: ET.Element, task: Task):
        file_name = uses.get('file')
        size = int(uses.get('size', '0'))
        link_type = uses.get('link', 'input').lower()
        if link_type == 'output':
            task.output_files[file_name] = size
            self._file_producers[file_name] = task.id
            self._file_sizes[file_name] = size
        else:
            task.input_files[file_name] = size

    def _process_dependencies(self, child: ET.Element, dag: DAG):
        child_ref = child.get('ref')
        child_task = self._task_map[child_ref]

        for parent in child.findall('.//pegasus:parent', self.ns):
            parent_ref = parent.get('ref')
            parent_task = self._task_map[parent_ref]

            transfer_size = sum(
                size for filename, size in parent_task.output_files.items()
                if filename in child_task.input_files
            )
            if transfer_size == 0 and parent_task.id >= 0 and child_task.id >= 0:
                transfer_size = 1000000
                print(f"Force set default transfer {parent_task.id}->{child_task.id} (1MB)")

            if child_task.id == dag.exit_id:
                transfer_size = 0

            dag.add_dependency(parent_task, child_task, transfer_size)

    def _link_orphan_tasks(self, dag: DAG):
        DEFAULT_FILE_SIZE = 1000000

        for task in [t for t in dag.tasks.values() if t.id >= 0]:
            has_real_pred = any(p >= 0 for p in dag.requiring.get(task.id, []))
            if not has_real_pred and dag.entry_id not in dag.requiring[task.id]:
                dag.add_dependency(dag.entry_task, task, DEFAULT_FILE_SIZE)

            has_real_succ = any(s >= 0 for s in dag.contributeTo.get(task.id, []))
            if not has_real_succ and dag.exit_id not in dag.contributeTo[task.id]:
                dag.add_dependency(task, dag.exit_task, 0)

    def _process_file_transfers(self, dag: DAG):
        print("\n=== File Transfer Debug ===")
        transfer_stats = defaultdict(list)
        for (src, dst), size in dag.file_transfers.items():
            transfer_stats[(src, dst)].append(size)
        print("File Transfer Examples (Top 5):")
        for (src, dst), sizes in list(transfer_stats.items())[:5]:
            print(f"  {src} -> {dst}: {sum(sizes)} bytes (Total {len(sizes)} files)")
        entry_transfers = sum(1 for (s, _) in dag.file_transfers if s == dag.entry_id)
        exit_transfers = sum(1 for (_, d) in dag.file_transfers if d == dag.exit_id)
        print(f"\nVirtual Node File Transfers:")
        print(f"  Files sent by Entry node: {entry_transfers}")
        print(f"  Files received by Exit node: {exit_transfers}")