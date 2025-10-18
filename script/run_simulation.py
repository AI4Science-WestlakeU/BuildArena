"""
Example Usage:
uv run script/run_simulation.py -d path/to/simulation_database.db -r
"""

import argparse
import os
import yaml
import shutil
import json

from scheduler.task_db import * 
from scheduler.scheduler import Scheduler

def load_levels_yaml(levels_file="levels.yaml"):
    """Load and parse the levels.yaml file to extract available categories and difficulty levels."""
    with open(levels_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse YAML documents (each level is a separate YAML document)
    levels = []
    for doc in yaml.safe_load_all(content):
        if doc:  # Skip empty documents
            levels.append(doc)
    
    return levels

def get_available_categories_and_levels(levels):
    """Extract unique categories and levels from the parsed levels data."""
    categories = set()
    difficulty_levels = set()
    
    for level_data in levels:
        if 'category' in level_data:
            categories.add(level_data['category'])
        if 'level' in level_data:
            difficulty_levels.add(level_data['level'])
    
    return sorted(list(categories)), sorted(list(difficulty_levels))

def find_level_task(levels, category, difficulty_level):
    """Find and return the task content for a specific category and difficulty level."""
    for level_data in levels:
        if (level_data.get('category') == category and 
            level_data.get('level') == difficulty_level):
            return level_data.get('task', '')
    
    return None

def init_simulation_db(
        original_db_path: str, 
        sim_db_path: str, 
        original_machine_dir: str, 
        sim_machine_dir: str, 
        max_workers: int,
        stage: str,
        proj_name_for_db: str,
        rfd_only: bool = True,
        ):
    # Copy original machine directory to simulation machine directory
    shutil.copytree(original_machine_dir, sim_machine_dir, dirs_exist_ok=True)

    # Duplicate the original db to the simulation db
    shutil.copy(original_db_path, sim_db_path)

    original_config = fetch_simulation_config(db_path=sim_db_path)
    config_dict = original_config.config
    global_config = original_config.global_config
    global_config['max_workers'] = max_workers

    # Remove the original config from simulation db
    remove_config(config_id=original_config.id, db_path=sim_db_path)
    insert_config(config=config_dict, global_config=global_config, db_path=sim_db_path)

    sim_config = fetch_simulation_config(db_path=sim_db_path)
    config_id = sim_config.id
    config_dict = sim_config.config
    goal = config_dict['project']['goal']

    # Fetch all plan tasks in the original db
    plan_tasks = fetch_tasks_by_stage(stage="plan", db_path=original_db_path)
    for plan_task in plan_tasks:
        plan_id = plan_task.id
        num_sub_structures = plan_task.sub_structure
        plan_file_path = plan_task.result_path.replace(".md", ".json") if plan_task.result_path else None
        plan_dict = json.load(open(plan_file_path, "r", encoding="utf-8")) if plan_file_path else None
        plan_content = "\n".join([plan_dict['overall_structure']['functionality'], plan_dict['overall_structure']['motion_control']]) if plan_dict else None
        task_content = f"The task instruction: \n{goal}\n\n---\n\n The plan of the machine structure: \n{plan_content}"
        if num_sub_structures > 1:
            # Find assemble tasks bind to the plan task
            assemble_task_ids = fetch_task_ids_by_plan(plan_id=plan_id, stage="assemble", db_path=original_db_path)
            for assemble_task_id in assemble_task_ids:
                machine_id = assemble_task_id
                insert_task(
                            name=f"{stage}_{machine_id}" if not check_task_exists(task_name=f"{stage}_{machine_id}", db_path=sim_db_path) else None,
                            stage=stage,
                            content=task_content,
                            db_path=sim_db_path,
                            bind_machine=machine_id,
                            bind_config=config_id,
                            bind_plan=plan_id,
                            project_name=proj_name_for_db,
                        )
        else:
            add_simulation_task(plan_id=plan_id, content=task_content, stage=stage, config_id=config_id, proj_name=proj_name_for_db, db_path=sim_db_path, rfd_only=rfd_only)

def main(original_db_path: str = None, rfd_only: bool = True, max_workers: int = 4, resume: bool = False):
    assert original_db_path, "db_path is required"
    
    original_proj_dir = os.path.dirname(original_db_path)
    proj_name = os.path.basename(original_proj_dir)
    category = proj_name.split("_")[0]
    level = proj_name.split("_")[1]
    stage = "simulation" if category != "transport" else "control"

    original_machine_dir = os.path.join(original_proj_dir, "machine")
    sim_proj_dir = os.path.join("datacache", f"{proj_name}_sim")
    sim_machine_dir = os.path.join(sim_proj_dir, "machine")
    
    proj_name_for_db = f"{proj_name}_sim"

    sim_db_path = os.path.join(sim_proj_dir, "simulation_database.db")

    if not resume:
        print("Initializing empty simulation db")
        init_simulation_db(
            original_db_path=original_db_path, 
            sim_db_path=sim_db_path, 
            original_machine_dir=original_machine_dir, 
            sim_machine_dir=sim_machine_dir, 
            max_workers=max_workers, 
            stage=stage, 
            proj_name_for_db=proj_name_for_db, 
            rfd_only=rfd_only)
    else: 
        assert os.path.exists(sim_db_path), f"Simulation db {sim_db_path} does not exist"
        print("Cleaning failed count to resume simulation db")
        clean_failed_count(db_path=sim_db_path)

    # Exclude non-simulation tasks in the simulation db
    print("Cleaning non-simulation tasks in simulation db")
    clean_tasks(sim_db_path)

    # Initialize scheduler with project settings
    if category == "transport":
        timeout = 14400  # 4 hours
    else:
        timeout = 5400  # 1.5 hours
    scheduler = Scheduler.from_db_path(sim_db_path, timeout=timeout)
    scheduler.run(break_on_complete=True)
    print(f"Simulation completed: {sim_db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run project tasks')
    parser.add_argument('--db', '-d', 
                      type=str, 
                      required=False,
                      help='Path to the task_database.db file')
    parser.add_argument('--todo', '-t', 
                      type=str, 
                      required=False,
                      help='Path to the todo.json file store the db_path to be simulated')
    parser.add_argument('--rfd', '-r', 
                      action='store_true', 
                      default=True,
                      required=False,
                      help='Only add simulation tasks for RFD machines')
    parser.add_argument('--max-workers', '-mw', 
                      type=int, 
                      required=False,
                      default=4,
                      help='Maximum number of workers to use for simulation')
    parser.add_argument('--resume', '-rs', 
                      action='store_true', 
                      required=False,
                      help='Resume the simulation from the simulation db of the given task db')
    # Parse arguments first to check for level-based mode
    args = parser.parse_args()
    if args.todo:
        with open(args.todo, 'r', encoding='utf-8') as f:
            todo_list = json.load(f)
        
        # Deduplicate the todo list
        todo_list = list(set(todo_list))

        # Remove non-existent db paths
        todo_list = [db_path for db_path in todo_list if os.path.exists(db_path)]

        # Save cleaned todo list
        clean_todo_path = args.todo.replace(".json", "_clean.json")
        with open(clean_todo_path, 'w', encoding='utf-8') as f:
            json.dump(todo_list, f, indent=2)
        
        assert len(todo_list) > 0, "Todo list is empty"
        print(f"Todo list: {todo_list}")
        # If support task is in the todo list, assert they all have same level
        if "support" in todo_list:
            assert len(set([db_path.split("_")[1] for db_path in todo_list])) == 1, "Support tasks must have same level"
    
    else:
        assert args.db
        todo_list = [args.db]

    # Check if the simulation db already exists
    filtered_todo_list = []
    for idx, db_path in enumerate(todo_list):
        proj_name = os.path.basename(os.path.dirname(db_path))
        sim_db_path = os.path.join(os.path.dirname(os.path.dirname(db_path)), f"{proj_name}_sim", "simulation_database.db")
        print(f"Simulation db path: {sim_db_path}")
        if os.path.exists(sim_db_path) and not args.resume:
            print(f"Simulation db {sim_db_path} already exists with no resume flag. \n\nAre you sure you want to start a new simulation? This will delete the existing simulation db. (Y/n)")
            if input() == "Y":
                filtered_todo_list.append(db_path)
        else:
            filtered_todo_list.append(db_path)

    # Run simulation
    print(f"Todo list: {filtered_todo_list}")
    if len(filtered_todo_list) == 0:
        print("No simulation to run")
        exit()
    for db_path in todo_list:
        main(original_db_path=db_path, rfd_only=args.rfd, max_workers=args.max_workers, resume=args.resume)
