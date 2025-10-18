from sqlmodel import Field, SQLModel, create_engine, Session, select, or_, and_, text
from typing import Optional, List, Tuple
from datetime import datetime
import random
import string
import os
import yaml
import json
import pdb
import shutil
from pathlib import Path
import re

# Default database path
DEFAULT_DB_PATH = "./datacache/default/task_database.db"
IGNORE_ERROR_INFO = ["parseerror", "jsondecodeerror", "valueerror: the machine",  "valueerror: no control"]


def normalize_yaml(filepath: str):
    """
    Reads a YAML file, normalizes the 'agents' field from a list to a nested dict keyed by 'name',
    and overwrites the file with the updated structure.
    
    Example conversion:
    agents:
      - name: planner
        system_message: "..."
    
    becomes:
    agents:
      planner:
        name: planner
        system_message: "..."
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and isinstance(data.get('agents'), list):
        new_agents = {}
        for agent in data['agents']:
            agent_name = agent.get('name')
            if agent_name:
                new_agents[agent_name] = agent
        data['agents'] = new_agents

        # Overwrite the file with the new format
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        
def generate_task_id():
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choices(characters, k=8))

def _normalize_path(p: str | Path, base: str | Path | None = None) -> Path:
    # Use regex to split the path into parts and then join them back together
    parts = re.split(r"[\\/]+", str(p))
    path = Path(*parts)
    if not path.is_absolute():
        base = Path(base) if base else Path.cwd()
        path = base / path
    return path.resolve()

def load_config(config_path):
    """Load project configuration from YAML file"""
    config_path = _normalize_path(config_path)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        config_path_str = str(config_path).replace("datacache", "datacache/data")
        config_path = Path(config_path_str)
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

class Config(SQLModel, table=True):
    id: str = Field(default_factory=generate_task_id, primary_key=True)
    file_path: str
    global_config_path: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    db_path: Optional[str] = DEFAULT_DB_PATH
    bind_task: Optional[str] = None
    pass_rate: float = 0.0
    average_cost: float = 0.0
    performance: Optional[str] = None
    status: str = "pending"
    debug: bool = False
    duration: Optional[float] = None
    token_input: int = 0
    token_output: int = 0
    
    @property
    def config(self) -> dict:
        return load_config(self.file_path)

    @property
    def global_config(self) -> dict:
        return load_config(self.global_config_path)

class Task(SQLModel, table=True):
    id: str = Field(default_factory=generate_task_id, primary_key=True)
    name: Optional[str] = None
    stage: Optional[str] = None
    status: str = "pending"
    content: Optional[str] = None
    parent_id: Optional[str] = Field(default=None, foreign_key="task.id")
    result_path: Optional[str] = None
    key_result: Optional[bool] = False
    error_info: Optional[str] = None
    db_path: Optional[str] = DEFAULT_DB_PATH
    project_name: Optional[str] = "default"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    retry_count: int = 0
    max_retries: int = 3
    bind_machine: Optional[str] = None
    bind_config: Optional[str] = None
    bind_plan: Optional[str] = Field(default=None, foreign_key="task.id")
    plan_content: Optional[str] = None
    sub_structure: Optional[int] = 1
    raise_objection: Optional[str] = None
    n_offspring: int = 1
    n_objection: int = 0
    duration: Optional[float] = None
    turn: int = 0
    token_input: int = 0
    token_output: int = 0

class Machine(SQLModel, table=True):
    id: str = Field(default_factory=generate_task_id, primary_key=True)
    name: Optional[str] = None
    description: Optional[str] = None
    status: str = "pending"
    sub_structure: Optional[int] = 1
    blueprint: Optional[str] = None
    file_path: Optional[str] = None
    approved: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    db_path: Optional[str] = DEFAULT_DB_PATH
    bind_task: Optional[str] = None
    bind_config: Optional[str] = None
    bind_plan: Optional[str] = None
    num_blocks: int = 0
    cost: int = 0
    registered: bool = False

def get_engine(db_path: Optional[str] = None):
    db_path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    # WAL mode
    with Session(engine) as session:
        session.exec(text("PRAGMA journal_mode=WAL;"))
    return engine

def init_db(db_path: Optional[str] = None):
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)
    
    # Clean up tasks with None name
    with Session(engine) as session:
        # Handle migration for new fields
        statement = select(Task)
        tasks = session.exec(statement).all()
        for task in tasks:
            if not hasattr(task, 'retry_count'):
                task.retry_count = 0
            if not hasattr(task, 'max_retries'):
                task.max_retries = 3
        session.commit()
        
        # Clean up invalid tasks
        statement = select(Task).where(Task.name.is_(None))
        invalid_tasks = session.exec(statement).all()
        if invalid_tasks:
            for task in invalid_tasks:
                session.delete(task)
            session.commit()
            print(f"Cleaned up {len(invalid_tasks)} redundant tasks")
    
    print(f"Database initialized at {db_path or DEFAULT_DB_PATH}")

def insert_config(config: dict, global_config: dict, bind_task: Optional[str] = None, db_path: Optional[str] = None) -> str:
    project_dir = os.path.dirname(db_path)
    config_dir = os.path.join(project_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_id = generate_task_id()
    file_path = os.path.join(config_dir, f"{config_id}.yaml")
    global_config_path = os.path.join(config_dir, f"global_config.yaml")
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    with open(global_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(global_config, f, sort_keys=False, allow_unicode=True)
    normalize_yaml(file_path)
    normalize_yaml(global_config_path)
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        config = Config(id=config_id, file_path=file_path, global_config_path=global_config_path, db_path=db_path, bind_task=bind_task, status="pending")
        session.add(config)
        session.commit()
    
    return config.id

def check_config_completed(config_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    engine = get_engine(db_path)
    with Session(engine) as session:
        config = session.get(Config, config_id)
        if config.status == "completed":
            return {
                "pass_rate": config.pass_rate,
                "average_cost": config.average_cost,
                "performance": config.performance,
                "duration": config.duration
            }
        else:
            return None

def insert_machine(id: str, 
                   file_path: str, 
                   bind_config: str, 
                   bind_task: str, 
                   bind_plan: str,
                   sub_structure: str, 
                   name: Optional[str] = None, 
                   description: Optional[str] = None, 
                   cost: int = 0, 
                   num_blocks: int = 0, 
                   blueprint: Optional[str] = None,
                   db_path: Optional[str] = None):
    if id is None:
        print(f"Skip inserting machine.")
        return
    engine = get_engine(db_path)
    with Session(engine) as session:
        machine = Machine(id=id, 
                          file_path=file_path, 
                          bind_config=bind_config, 
                          bind_task=bind_task, 
                          bind_plan=bind_plan,
                          sub_structure=sub_structure, 
                          name=name, 
                          description=description, 
                          cost=cost, 
                          num_blocks=num_blocks, 
                          blueprint=blueprint,
                          db_path=db_path)
        task = session.get(Task, bind_task)
        # If the task has raised objection, mark the machine as objected
        if "_rfd" in name:
            parent_machine = session.get(Machine, name.replace("_rfd", ""))
            parent_machine.status = "objected"
            parent_machine.approved = False
            parent_machine.updated_at = datetime.now().isoformat()
            session.add(parent_machine)
            session.commit()
        if task.raise_objection:
            machine.status = "objected"
            machine.approved = False
            machine.updated_at = datetime.now().isoformat()
        session.add(machine)
        session.commit()

def register_machine(machine_id: str, caption: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        machine = session.get(Machine, machine_id)
        if machine:
            machine.registered = True
            machine.caption = caption
            session.add(machine)
        
def get_config(id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        config = session.get(Config, id)
        return config
    
def update_config(id: str, pass_rate: float, average_cost: float, performance: Optional[str] = None, token_input: int = 0, token_output: int = 0, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        config = session.get(Config, id)
        config.pass_rate = pass_rate
        config.average_cost = average_cost
        config.performance = performance
        config.updated_at = datetime.now().isoformat()
        config.status = "completed"
        config.duration = (datetime.fromisoformat(config.updated_at) - datetime.fromisoformat(config.created_at)).total_seconds() / 60.0
        config.token_input = token_input
        config.token_output = token_output
        session.add(config)
        session.commit()
        
def insert_task(name: str, 
                stage: str, 
                content: str, 
                bind_config: Optional[str] = None,
                bind_plan: Optional[str] = None,
                parent_id: Optional[str] = None, 
                db_path: Optional[str] = None,
                project_name: Optional[str] = None,
                sub_structure: Optional[str] = None, 
                bind_machine: Optional[str] = None, 
                plan_content: Optional[str] = None,
                n_offspring: int = 1
                ):
    if name is None:
        print(f"Skip inserting task.")
        return
    
    # Handle insert tasks from plan situation
    bind_plan = bind_plan or parent_id
    engine = get_engine(db_path)
    now = datetime.now().isoformat()
    task = Task(
        name=name,
        stage=stage,
        content=content,
        bind_config=bind_config,
        bind_plan=bind_plan,
        parent_id=parent_id,
        created_at=now,
        updated_at=now,
        db_path=db_path,
        project_name=project_name,
        sub_structure=sub_structure,
        bind_machine=bind_machine,
        plan_content=plan_content,
        n_offspring=n_offspring
    )
    with Session(engine) as session:
        session.add(task)
        session.commit()
        print(f"Task '{name}' inserted with ID {task.id}")

def get_task(task_id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        task = session.get(Task, task_id)
        return task

def get_pending_ready_tasks(db_path: Optional[str] = None) -> List[Task]:
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Task).where(
            Task.status == "pending",
            ((Task.stage == "plan") | (Task.stage != "plan" and Task.parent_id.is_not(None)))
        )
        results = session.exec(statement).all()
    return results

def get_completed_plan_tasks(db_path: Optional[str] = None) -> List[Task]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Task).where(
            and_(Task.status == "completed", Task.stage == "plan")
        )
        results = session.exec(statement).all()
    return results

def fetch_machine_ids_by_plan(plan_id: str, 
                          status: str = "completed", 
                          approved: bool = True,
                          registered: bool = True,
                          sub_structure: int = 1,
                          db_path: Optional[str] = None) -> List[str]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Machine).where(
            and_(Machine.bind_plan == plan_id, 
                 Machine.status == status,
                 Machine.approved == approved,
                 Machine.registered == registered,
                 Machine.sub_structure == sub_structure)
            )
        machines = session.exec(statement).all()
        return [machine.id for machine in machines]

def fetch_task_ids_by_plan(plan_id: str, stage: str = "build", db_path: Optional[str] = None) -> List[str]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Task).where(and_(Task.bind_plan == plan_id, Task.stage == stage))
        tasks = session.exec(statement).all()
        return [task.id for task in tasks]

def fetch_verify_results_by_machine(machine_id: str, db_path: Optional[str] = None) -> List[bool]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Task).where(and_(Task.bind_machine == machine_id, Task.stage == "verify", Task.status == "completed"))
        tasks = session.exec(statement).all()
        return [task.key_result for task in tasks]

def fetch_and_mark_tasks(limit: int, db_path: Optional[str]) -> List[Task]:
    """Fetch and mark non-simulation tasks that are pending or failed"""
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:

        statement = select(Task).where(
                or_(Task.status == "pending", Task.status == "failed")
            )
        tasks = session.exec(statement).all()
        
        tasks = [task for task in tasks if task.retry_count < task.max_retries and task.stage != "simulation"][:limit]

        for task in tasks:
            task.status = "processing"
            task.created_at = datetime.now().isoformat()
        session.commit()

        return tasks

def fetch_and_mark_one_simulation_task(limit: 1, db_path: Optional[str]) -> Optional[List[Task]]:
    """Fetch and mark simulation tasks that are pending or failed"""
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Task).where(or_(Task.status == "pending", Task.status == "failed"))
        tasks = session.exec(statement).all()
        if len(tasks) == 0:
            return None
        tasks = [task for task in tasks if task.retry_count < task.max_retries and task.stage == "simulation"][:limit]
        for task in tasks:
            task.status = "processing"
            task.created_at = datetime.now().isoformat()
        session.commit()
        return tasks[:limit]

def fetch_and_mark_config(db_path: Optional[str]) -> List[Config]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Config).where(Config.status == "pending")
        configs = session.exec(statement).all()
        for config in configs:
            config.status = "processing"
            config.updated_at = datetime.now().isoformat()
        session.commit()
        return configs
    
def fetch_unfinished_config(db_path: Optional[str]) -> List[Config]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Config).where(Config.status == "processing")
        configs = session.exec(statement).all()
        return configs

def fetch_machines(config_id: str, db_path: Optional[str]) -> List[Machine]:
    """Fetch all machines bound to the given config id."""
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Machine).where(Machine.bind_config == config_id)
        machines = session.exec(statement).all()
        return machines

def get_total_token_usage_by_config(config_id: str, db_path: Optional[str] = None) -> Tuple[int, int]:
    """
    Returns the sum of token_input and token_output for all tasks bound to the given config id.
    """
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Task).where(Task.bind_config == config_id)
        tasks = session.exec(statement).all()
        total_input = sum(task.token_input or 0 for task in tasks)
        total_output = sum(task.token_output or 0 for task in tasks)
        return total_input, total_output

def mark_task_completed(task_id: str, result_path: Optional[str] = None, key_result: Optional[str] = None, token_input: int = 0, token_output: int = 0, turn: int = 0, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.status = "completed"
            task.updated_at = datetime.now().isoformat()
            task.result_path = result_path
            task.key_result = key_result
            task.duration = (datetime.fromisoformat(task.updated_at) - datetime.fromisoformat(task.created_at)).total_seconds() / 60.0
            task.token_input = token_input
            task.token_output = token_output
            task.turn = turn
            session.add(task)
            session.commit()

def mark_task_processing(task_id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.status = "processing"
            task.updated_at = datetime.now().isoformat()
            session.add(task)
            session.commit()
            
def mark_task_failed(task_id: str, error_info: Optional[str] = None, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.status = "failed"
            task.updated_at = datetime.now().isoformat()
            task.error_info = error_info
            # Increment retry count when task fails
            task.retry_count += 1
            session.add(task)
            session.commit()


def check_task_exists(task_name: str, config_id: Optional[str] = None, db_path: Optional[str] = None) -> bool:
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Task)
        if config_id:
            statement = statement.where(Task.bind_config == config_id, Task.name == task_name)
        else:
            statement = statement.where(Task.name == task_name)
        exists = session.exec(statement).first() is not None
        return exists

def mark_machine_processing(machine_id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        machine = session.get(Machine, machine_id)
        if machine:
            machine.status = "processing"
            machine.updated_at = datetime.now().isoformat()
            session.add(machine)
            session.commit()
            
def mark_machine_completed(machine_id: str, approved: bool = False, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        machine = session.get(Machine, machine_id)
        if machine:
            machine.status = "completed"
            machine.updated_at = datetime.now().isoformat()
            machine.approved = approved
            session.add(machine)
            session.commit()
            
def mark_machine_unverified(machine_id: str, approved: bool = False, db_path: Optional[str] = None):
    """
    Mark a machine as unverified, which means the machine has not been verified by the verifier.
    """
    engine = get_engine(db_path)
    with Session(engine) as session:
        machine = session.get(Machine, machine_id)
        if machine:
            machine.status = "unverified"
            machine.updated_at = datetime.now().isoformat()
            machine.approved = approved
            session.add(machine)
            session.commit()
            
def get_machine(machine_id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        machine = session.get(Machine, machine_id)
        return machine
    
def check_machine_exists(machine_id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Machine).where(Machine.id == machine_id)
        exists = session.exec(statement).first() is not None
        return exists

def get_machine_dirs_by_config(config_id: str, db_path: Optional[str] = None) -> List[Tuple[bool, str]]:
    """
    Return a list of build directories for all machines bound to the given config_id.
    Each directory is datacache/{project_name}/build/{machine.id}, where project_name is from the Task bound to the machine.
    Only include directories that exist on disk.
    """
    machines = fetch_machines(config_id, db_path)
    if not machines:
        return []
    engine = get_engine(db_path)
    build_dirs: List[Tuple[bool, str]] = []
    with Session(engine, expire_on_commit=False) as session:
        for machine in machines:
            project_name = os.path.basename(os.path.dirname(db_path))
            build_dir = os.path.join("datacache", project_name, "machine", machine.id)
            if os.path.isdir(build_dir):
                build_dirs.append((machine.approved, build_dir))
    return build_dirs

def update_plan_sub_structures(task_id: str, num_sub_structures: int, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.sub_structure = num_sub_structures
            session.add(task)
            session.commit()
            
def update_task_machine(task_id: str, bind_machine: Optional[str] = None, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.bind_machine = bind_machine
            session.add(task)
            session.commit()
            
def add_task_objection(task_id: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.n_objection += 1
            if task.n_objection == task.n_offspring:
                task.status = "objected"
                task.updated_at = datetime.now().isoformat()
                task.error_info = "Objection count exceeds the number of offspring"
            session.add(task)
            session.commit()
            
def update_task_raise_objection(task_id: str, raise_objection: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.raise_objection = raise_objection
            session.add(task)
            session.commit()

def fetch_pending_machine_number(db_path: Optional[str] = None) -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Machine).where(Machine.status == "pending")
        machines = session.exec(statement).all()
        return len(machines)

def fetch_error_simulation_number(db_path: Optional[str] = None) -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Task).where(Task.status == "error")
        tasks = session.exec(statement).all()
        return len(tasks)
    
def fetch_pending_task_number(db_path: Optional[str] = None) -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Task).where(Task.status == "pending")
        tasks = session.exec(statement).all()
        return len(tasks)

def get_plan_sub_structures(plan_id: str, db_path: Optional[str] = None) -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        plan = session.get(Task, plan_id)
        return plan.sub_structure if plan else 1

def clean_failed_count(db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Task).where(Task.result_path != None)
        tasks = session.exec(statement).all()
        for task in tasks:
            task.status = "completed"
        session.commit()
    with Session(engine) as session:
        statement = select(Task).where(or_(Task.status == "failed", Task.status == "processing"))
        tasks = session.exec(statement).all()
        for task in tasks:
            if task.status == "failed":
                if any(error_info in task.error_info.lower() for error_info in IGNORE_ERROR_INFO):
                    continue
            task.retry_count = 0
            task.status = "pending"
        session.commit()

def check_simulation_db(db_path: Optional[str] = None):
    db_name = os.path.basename(db_path)
    assert "simulation" in db_name or "ablation" in db_name, "Only simulation db is supported for cleaning tasks"

def clean_tasks(db_path: Optional[str] = None):
    check_simulation_db(db_path)
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Task).where(and_(Task.stage != "simulation", Task.stage != "control"))
        tasks = session.exec(statement).all()
        for task in tasks:
            session.delete(task)
        session.commit()

# Add simulation task for each machine
def add_simulation_task(plan_id: str, content: str, stage: str, config_id: str, proj_name: str, db_path: str, rfd_only: bool = False):
    check_simulation_db(db_path)
    engine = get_engine(db_path)
    with Session(engine) as session:
        statement = select(Machine).where(Machine.bind_plan == plan_id)
        machines = session.exec(statement).all()
        machine_ids = [machine.id for machine in machines]
        for machine in machines:
            # Refresh the machine db path
            machine.db_path = db_path
            machine_id = machine.id
            rfd_id = f"{machine_id}_rfd"
            if rfd_id in machine_ids and rfd_only:
                machine_id = rfd_id
            else:
                machine_id = machine.id
            insert_task(
                name=f"{stage}_{machine_id}" if not check_task_exists(task_name=f"{stage}_{machine_id}", db_path=db_path) else None,
                stage=stage,
                content=content,
                db_path=db_path,
                bind_machine=machine_id,
                bind_plan=plan_id,
                bind_config=config_id,
                project_name=proj_name,
            )

def fetch_simulation_config(db_path: str) -> Config:
    # check_simulation_db(db_path)
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Config)
        config = session.exec(statement).first()
        config.status = "simulation"
        config.updated_at = datetime.now().isoformat()
        session.add(config)
        session.commit()
        return config

def mark_simulation_task(task_id: str, status: str, db_path: Optional[str] = None):
    engine = get_engine(db_path)
    with Session(engine) as session:
        task = session.get(Task, task_id)
        if task:
            task.status = status
            task.updated_at = datetime.now().isoformat()
            session.add(task)
            session.commit()

def fetch_tasks_by_stage(stage: str, db_path: Optional[str] = None) -> List[Task]:
    engine = get_engine(db_path)
    with Session(engine, expire_on_commit=False) as session:
        statement = select(Task).where(Task.stage == stage)
        tasks = session.exec(statement).all()
        return tasks

def remove_config(config_id: str, db_path: Optional[str] = None):
    check_simulation_db(db_path)
    engine = get_engine(db_path)
    with Session(engine) as session:
        session.delete(session.get(Config, config_id))
        session.commit()
