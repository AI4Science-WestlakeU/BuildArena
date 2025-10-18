import asyncio
import time
import os
import pdb
import csv
import json
import numpy as np
from typing import Optional, Callable, Dict

from scheduler.task_db import *
from spatial.agent import MultiAgents, ProcessContext

from simulation.simulation_transport import main as simulation_transport
from simulation.simulation_lift import main as simulation_lift
from simulation.simulation_support import main as simulation_support

# Initialize agents
agents = MultiAgents(verbose=False)

async def run_plan(task: Task, global_config: dict) -> ProcessContext:
    plan_context = await agents.plan(task)
    plan_structures: Dict[str, Dict[str, str]] = plan_context.result['sub_structures']
    for sub_index, sub_structure in enumerate(plan_structures.values()):
        sub_name = sub_index + 1
        content =  f"The plan of the specific sub-structure {sub_name}: {str(sub_structure)}"
        insert_task(name=f"Successor draft task {sub_name} of {task.id}", 
                    stage="draft", 
                    content=content, 
                    parent_id=task.id, 
                    db_path=task.db_path,
                    project_name=task.project_name, 
                    bind_config=task.bind_config, 
                    bind_plan=task.bind_plan,
                    sub_structure=sub_name, 
                    bind_machine=None,
                    plan_content=str(sub_structure)
                    )

    update_plan_sub_structures(task_id=task.id, num_sub_structures=len(plan_structures.keys()), db_path=task.db_path)

    return plan_context

async def run_draft(task: Task, global_config: dict) -> ProcessContext:
    draft_context = await agents.draft(task)
    blueprint = f"<blueprint>\n{draft_context.result}\n<end blueprint>"
    objection = draft_context.objection
    # If the task has raised objection, return the draft context and do not create any offspring
    if objection:
        return draft_context
    
    insert_task(name=f"Successor build task {task.sub_structure} of {task.id}", 
                    stage="build", 
                    content=blueprint, 
                    bind_config=task.bind_config,
                    bind_plan=task.bind_plan,
                    parent_id=task.id,
                    db_path=task.db_path,
                    project_name=task.project_name,
                    sub_structure=task.sub_structure, 
                    bind_machine=None,
                    plan_content=task.plan_content)
    
    return draft_context

async def run_build(task: Task, global_config: dict) -> ProcessContext:
    blueprint = task.content
    build_context = await agents.build(task)
    build_result = build_context.result
    objection = build_context.objection
    
    file_path = os.path.join(os.path.dirname(task.db_path), "machine", task.id, f"{task.id}.json")
    machine_state = build_result['result']
    spinful = build_result['has_spinful']
    machine_content = blueprint
    machine_id = task.id
    plan_content = task.plan_content

    update_task_machine(task_id=task.id, bind_machine=machine_id, db_path=task.db_path)

    # Add machine to database
    insert_machine(id=machine_id if not check_machine_exists(machine_id, task.db_path) else None, 
                file_path=file_path, 
                bind_config=task.bind_config, 
                bind_plan=task.bind_plan,
                db_path=task.db_path,
                bind_task=task.id, 
                sub_structure=task.sub_structure, 
                name=task.id, 
                description=build_result['result'], 
                cost=build_result['cost'], 
                num_blocks=build_result['num_blocks'], 
                blueprint=blueprint,
                )
    
    if objection:
        # The machine has been built with the Agent objection, do not need to create any offspring
        return build_context

    if not spinful:
        mark_machine_completed(machine_id=machine_id, approved=False, db_path=task.db_path)

    else:
        # Initialize a refining task from machine
        machine_content += f"<build result>\n{machine_state}\n<end build result>"
        refine_content = f"The machine has been built, now please carefully review the rotation of the rotatable blocks, refine the machine to meet the design requirement./n{machine_content}"
        insert_task(
            name=f"{machine_id}_rfd" if not check_task_exists(task_name=f"{machine_id}_rfd", db_path=task.db_path) else None,
            stage="refine",
            content=refine_content,
            db_path=task.db_path,
            project_name=task.project_name,
            parent_id=task.id,
            bind_config=task.bind_config, 
            bind_plan=task.bind_plan,
            sub_structure=task.sub_structure, 
            bind_machine=f"{machine_id}_rfd", 
            plan_content=plan_content
        )

    return build_context

async def run_refine(task: Task, global_config: dict) -> ProcessContext:
    # Remove the build result from the content of the refine task
    blueprint = task.content.split("<blueprint>")[-1].split("<end blueprint>")[0]
    refine_context = await agents.build(task, refine=True)
    refine_result = refine_context.result
    file_path = os.path.join(os.path.dirname(task.db_path), "machine", task.id, f"{task.id}.json")
    machine_id = task.bind_machine  # In format of {machine_id}_rfd
    objection = refine_context.objection
       
    insert_machine(id=machine_id if not check_machine_exists(machine_id, task.db_path) else None, 
                   file_path=file_path, 
                   bind_config=task.bind_config, 
                   bind_task=task.id, 
                   bind_plan=task.bind_plan,
                   sub_structure=task.sub_structure, 
                   name=machine_id, 
                   description=refine_result['result'], 
                   cost=refine_result['cost'], 
                   num_blocks=refine_result['num_blocks'], 
                   db_path=task.db_path, 
                   blueprint=blueprint,
                   )
    if objection:
        # The machine has been refined with the Agent objection, do not need to create any offspring
        return refine_context
    
    else:
        # The machine has been refined, mark it as unverified
        mark_machine_unverified(machine_id=machine_id, approved=True, db_path=task.db_path)

    return refine_context

async def run_assemble(task: Task, global_config: dict) -> ProcessContext:
    assemble_context = await agents.build(task, assemble=True)
    assemble_result = assemble_context.result
    file_path = os.path.join(os.path.dirname(task.db_path), "machine", task.id, f"{task.id}.json")
    machine_id = task.id
    plan_content = task.plan_content
    insert_machine(id=machine_id if not check_machine_exists(machine_id, task.db_path) else None, 
                   file_path=file_path, 
                   bind_config=task.bind_config, 
                   bind_task=task.id, 
                   bind_plan=task.bind_plan,
                   sub_structure=task.sub_structure, 
                   name=machine_id, 
                   description=assemble_result['result'], 
                   cost=assemble_result['cost'], 
                   num_blocks=assemble_result['num_blocks'], 
                   db_path=task.db_path, 
                   blueprint=plan_content,
                   )
    objection = assemble_context.objection
    if objection:
        return assemble_context
    
    mark_machine_unverified(machine_id=machine_id, approved=True, db_path=task.db_path)
    return assemble_context

async def run_control(task: Task, global_config: dict) -> ProcessContext:
    """Transport specific control task"""
    parent_id = task.parent_id
    machine_id = task.bind_machine
    if parent_id:
        # Simulation task is the parent of the control task
        parent_task = get_task(task_id=parent_id, db_path=task.db_path)
        sim_data_path = parent_task.result_path  # csv file path
        if not sim_data_path:
            raise ValueError("Simulation data not found")
        gen = int(sim_data_path.split("_")[-1].split(".")[0])
        # Previous control task
        previous_control_task = get_task(task_id=parent_task.parent_id, db_path=task.db_path)
        previous_control_messages_path = previous_control_task.result_path.replace(".md", "_messages.json")  # JSON file path
        # Read in JSON file
        with open(previous_control_messages_path, "r", encoding="utf-8") as f:
            previous_control_messages = json.load(f)
        # Read in csv file
        with open(sim_data_path, "r", encoding='utf-8') as f:
            csv_data = csv.reader(f)
            csv_data = list(csv_data)
        
        valid_data = []
        for row in csv_data:
            if len(row) == 5:
                if float(row[0]) % 1 < 0.001 and not row[-1].strip().endswith(("-", ".", ",")):
                    try:
                        x_pos = float(row[2])
                        y_pos = float(row[4])
                        z_pos = float(row[3])
                        valid_data.append([row[0], row[1], x_pos, y_pos, z_pos])
                    except (ValueError, IndexError):
                        # Skip the rows that cannot be converted
                        continue
        csv_data = valid_data
        # Determine if the machine is not moving at all
        positions = np.array([row[2:5] for row in csv_data], dtype=float)    
        pos_mean = np.mean(positions, axis=0)
        if np.all(np.abs(positions - pos_mean) < 1) and gen > 1:
            raise ValueError("The machine is not moving at all")
        # Take int digit rows as simulation feedback
        init_row = csv_data[0]
        simulation_feedback = ["The simulation trajectory (time, Block ID, x, y, z): "]
        for row in csv_data:
            row_id = row[1]
            row_time = float(row[0])
            row = [(float(row[i]) - float(init_row[i])) for i in [2, 3, 4]]  # x, z, y
            simulation_feedback.append(f"t={row_time:.1f}, ID={row_id}, x={row[0]:.2f}, y={row[2]:.2f}, z={row[1]:.2f}")  # time, ID, x, y, z
        simulation_feedback = "\n".join(simulation_feedback).replace("ID_Ballast", "Cargo Load")
        revision = (previous_control_messages, simulation_feedback)
        gen += 1
    else:
        sim_data_path = None
        gen = 1
        revision = None
        
    control_context = await agents.control(task=task, revision=revision, gen=gen)
    control_path = control_context.result
    gen = int(control_path.split("_")[-1].split(".")[0])
    insert_task(
                name=f"simulation_{machine_id}_{gen}" if not check_task_exists(task_name=f"simulation_{machine_id}_{gen}", db_path=task.db_path) else None,
                stage="simulation",
                content=control_path,
                db_path=task.db_path,
                bind_machine=machine_id,
                bind_config=task.bind_config,
                bind_plan=task.bind_plan,
                project_name=task.project_name,
                plan_content=task.content,
                parent_id=task.id,
            )
    return control_context

async def run_simulation(task: Task, global_config: dict) -> ProcessContext:
    await asyncio.sleep(1)
    machine_id = task.bind_machine
    db_path = task.db_path
    proj_name = task.project_name
    category = proj_name.split("_")[0]
    level = proj_name.split("_")[1]
    machine_json_path = os.path.join(os.path.dirname(db_path), "machine", machine_id, f"{machine_id}.json")
    simulation_context = f"Category: {category}, Level: {level}, Machine ID: {machine_id}, Project Name: {proj_name}"
    
    if category == "transport":
        control_path = task.content
        csv_file = simulation_transport(machine_json=machine_json_path, level=level, control_path=control_path)
        if not os.path.exists(csv_file):
            raise ValueError(f"Simulation data {csv_file} not saved")
        gen = int(control_path.split("_")[-1].split(".")[0])
        if gen < 3:
            insert_task(name=f"control_{machine_id}_{gen+1}" if not check_task_exists(task_name=f"control_{machine_id}_{gen+1}", db_path=db_path) else None,
                        stage="control",
                        content=task.plan_content,
                        db_path=db_path,
                        bind_machine=machine_id,
                        bind_config=task.bind_config,
                        bind_plan=task.bind_plan,
                        project_name=proj_name,
                        parent_id=task.id)

    elif category == "lift":
        csv_file = simulation_lift(machine_json=machine_json_path, level=level)
    elif category == "support":
        csv_file = simulation_support(machine_json=machine_json_path, level=level)
    else:
        raise ValueError(f"Category {category} not supported")
    
    # Check if the csv_file is empty
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        if len(next(reader)) == 0:
            mark_simulation_task(task_id=task.id, status="error", db_path=db_path)
            raise ValueError(f"CSV file {csv_file} is empty, simulation failed")
    # Check if the csv_file has lua error information
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        if "lua" in ",".join(next(reader)).lower():
            mark_simulation_task(task_id=task.id, status="error", db_path=db_path)
            raise ValueError(f"CSV file {csv_file} has Lua error, simulation failed")
    
    mark_simulation_task(task_id=task.id, status="simulated", db_path=db_path)
        
    context = ProcessContext(context=[simulation_context], 
                                 result=csv_file, 
                                 token_input=0, 
                                 token_output=0, 
                                 turn=0,
                                 objection=None)
    return context

runners: Dict[str, Callable] = {
    "plan": run_plan,
    "draft": run_draft,
    "build": run_build,
    "assemble": run_assemble,
    "control": run_control,
    "refine": run_refine, 
    "simulation": run_simulation,
}

