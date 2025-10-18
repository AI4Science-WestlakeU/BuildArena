import os
import json
import time
import multiprocessing
import argparse
from typing import Dict, List

from scheduler.task_db import *
from scheduler.worker import worker_process

class PlanManager:
    def __init__(self, plan_id: str, config_id: str, project_name: str, db_path: str):
        self.plan_id = plan_id
        self.config_id = config_id
        self.project_name = project_name
        self.db_path = db_path
        self.plan_file_path = os.path.join(os.path.dirname(db_path), 'plan', f'plan_{plan_id}.json')
        self.plan: Dict[str, Dict[str, str] | str] = json.load(open(self.plan_file_path, "r", encoding="utf-8"))
        self.sub_structures = self.plan['sub_structures']
        self.num_sub_structures = get_plan_sub_structures(self.plan_id, self.db_path)
        self.ready_machine_ids: Dict[int, List[str]] = {}
        self.pending_machine_ids: Dict[int, List[str]] = {}
        self.processing_machine_ids: Dict[int, List[str]] = {}
    
    def check_satisfied(self):
        if self.num_sub_structures == 1:
            return True
        for key, value in self.sub_structures.items():
            sub_structure_id = int(key.split('_')[-1])
            ready_machine_ids = fetch_machine_ids_by_plan(plan_id=self.plan_id, 
                                                    sub_structure=sub_structure_id, 
                                                    status="completed",
                                                    approved=True,
                                                    registered=False,
                                                    db_path=self.db_path)
            pending_machine_ids = fetch_machine_ids_by_plan(plan_id=self.plan_id, 
                                                    sub_structure=sub_structure_id, 
                                                    status="pending",
                                                    approved=True,
                                                    registered=False,
                                                    db_path=self.db_path)
            processing_machine_ids = fetch_machine_ids_by_plan(plan_id=self.plan_id, 
                                                    sub_structure=sub_structure_id, 
                                                    status="processing",
                                                    approved=True,
                                                    registered=False,
                                                    db_path=self.db_path)
            self.ready_machine_ids[sub_structure_id] = ready_machine_ids
            self.pending_machine_ids[sub_structure_id] = pending_machine_ids
            self.processing_machine_ids[sub_structure_id] = processing_machine_ids
        
        if (all(len(sub_structure_ids) == 0 for sub_structure_ids in self.pending_machine_ids.values()) 
            and all(len(sub_structure_ids) == 0 for sub_structure_ids in self.processing_machine_ids.values())
            and all(len(sub_structure_ids) > 0 for sub_structure_ids in self.ready_machine_ids.values())):
            return True
        return False
    
    def insert_assemble_task(self):
        if self.check_satisfied() and self.num_sub_structures > 1 and not check_task_exists(f"assemble_{self.plan_id}", db_path=self.db_path):
            insert_task(name=f"assemble_{self.plan_id}",
                        stage="assemble",
                        content=self.assemble_content,
                        db_path=self.db_path,
                        project_name=self.project_name,
                        bind_plan=self.plan_id, 
                        bind_config=self.config_id
                        )

    @property
    def assemble_content(self):
        prompt = [f"""
        Assemble the machines following the plan {self.plan['overall_structure']}. 
        Using at least one machine from each sub-structure: 
        """]
        for key, value in self.sub_structures.items():
            ready_machine_ids = self.ready_machine_ids[int(key.split('_')[-1])]
            machines = [get_machine(ready_machine_id, self.db_path) for ready_machine_id in ready_machine_ids]
            candidates_prompt = [f"""   
                                 <candidate_{index + 1}>
                                 machine ID: {machine.id}
                                 machine description: {machine.description}
                                 </candidate_{index + 1}>
                                 """ for index, machine in enumerate(machines)]
            sub_prompt = f"""
            <sub_structure_{key}>
            {value}
            candidates: {"\n".join(candidates_prompt)}
            </sub_structure_{key}>
            """
            prompt.append(sub_prompt)

        return "\n".join(prompt)

class Scheduler:
    def __init__(self, config: dict, db_path: str, timeout: int | None = None):
        self.max_workers = config['max_workers']
        self.active_processes = []
        self.db_path = db_path
        self.mode = "sampling" if "simulation" not in os.path.basename(db_path) else "simulation"
        self.global_config = config
        self.plan_managers: Dict[str, PlanManager] = {}
        self.timeout = timeout
        self.start_time = time.time()
        print("Scheduler initialized, DB path: ", self.db_path)
        
    @classmethod
    def from_db_path(cls, db_path: str, timeout: int | None = None):
        global_config = load_config(os.path.join(os.path.dirname(db_path), "configs", "global_config.yaml"))
        return cls(global_config, db_path, timeout)
    
    def run(self, break_on_complete=True):
        self.start_time = time.time()
        try:
            while True:
                if self.timeout and time.time() - self.start_time > self.timeout:
                    print(f"Timeout after {self.timeout} seconds, exiting...")
                    break
                
                # Clean up finished processes
                self.active_processes = [p for p in self.active_processes if p.is_alive()]
                active_tasks = [p.name for p in self.active_processes if not p.name.startswith("simulation")]
                active_simulation = [p.name for p in self.active_processes if p.name.startswith("simulation")]
                print(f"Active processes: {active_tasks}")
                print(f"Active simulation: {active_simulation}")
                print(f"DB: {self.db_path}")
                
                # Check if there are any pending configs
                configs = fetch_and_mark_config(db_path=self.db_path)
                if len(configs) > 0:
                    for config in configs:
                        # Initialize planning tasks from configuration
                        for i in range(self.global_config['n_sample']):
                            insert_task(
                                name=f"plan_{i}" if not check_task_exists(f"plan_{i}", config.id, self.db_path) else None,
                                stage="plan",
                                content=self.global_config['goal'],
                                db_path=self.db_path,
                                project_name=self.global_config['name'],
                                bind_config=config.id
                            )
                
                # Check for completed plan tasks and initialize plan manager
                completed_plan_tasks = get_completed_plan_tasks(self.db_path)
                if len(completed_plan_tasks) > 0:
                    for task in completed_plan_tasks:
                        if task.id not in self.plan_managers:
                            plan_manager = PlanManager(task.id, task.bind_config, self.global_config['name'], self.db_path)
                            self.plan_managers[task.id] = plan_manager    
                
                # Check if any plan manager is satisfied
                if len(self.plan_managers.values()) > 0:
                    for plan_manager in self.plan_managers.values():
                        plan_manager.insert_assemble_task()                 
                
                # Fetch all pending tasks and fill up worker slots
                available_slots = self.max_workers - len(active_tasks)
                if available_slots > 0:
                    tasks = fetch_and_mark_tasks(limit=available_slots, db_path=self.db_path)
                    for task in tasks[:available_slots]:
                        p = multiprocessing.Process(target=worker_process, args=(task,), name=f"{task.stage}_{task.id}")
                        p.start()
                        self.active_processes.append(p)
                if len(active_simulation) == 0:
                    simulation_tasks = fetch_and_mark_one_simulation_task(limit=1, db_path=self.db_path)
                    if simulation_tasks:
                        simulation_tasks = simulation_tasks[0]
                        p = multiprocessing.Process(target=worker_process, args=(simulation_tasks,), name=f"simulation_{simulation_tasks.id}")
                        p.start()
                        self.active_processes.append(p)

                time.sleep(5)
                pending_tasks = fetch_pending_task_number(db_path=self.db_path)
                pending_machines = fetch_pending_machine_number(db_path=self.db_path) if self.mode == "sampling" else 0
                all_satisfied = all(pm.check_satisfied() for pm in self.plan_managers.values()) if len(self.plan_managers.values()) > 0 else True
                all_ended = (pending_tasks == 0 and 
                            len(self.active_processes) == 0 and 
                            pending_machines == 0)
                if all_ended and all_satisfied:
                    if self.mode == "sampling":
                        self.post_process()
                    if break_on_complete:
                        print("All tasks completed.")
                        break

                error_simulation = fetch_error_simulation_number(db_path=self.db_path)
                if error_simulation > 2:
                    print(f"Simulation malfunctioned, exiting...")
                    break
        except KeyboardInterrupt:
            print("Keyboard interrupt, exiting...")
            exit()
                
if __name__ == "__main__":
    # Resume from a db path
    parser = argparse.ArgumentParser(description='Scheduler')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the .db file')
    parser.add_argument('--timeout', type=int, required=False, help='Timeout in seconds')
    args = parser.parse_args()
    db_path = args.db_path
    clean_failed_count(db_path=db_path)
    scheduler = Scheduler.from_db_path(db_path, timeout=args.timeout)
    scheduler.run()