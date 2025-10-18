import os
import asyncio
import argparse
import json

from scheduler.runner import runners
from scheduler.task_db import *
from spatial.agent import ProcessContext
from autogen_agentchat.messages import BaseChatMessage

def init_loop():
    # Create a new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def save_context(context: List[BaseChatMessage], task: Task, output_dir: str):
    """
    Save the context to a markdown file.
    """
    output_path = os.path.join(output_dir, f"{task.stage}_{task.id}.md")
    output_list = []
    config = get_config(task.bind_config, task.db_path).config
    with open(output_path, "w", encoding="utf-8") as f:
        for message in context:
            source = message.source
            model = config['agents'][message.source]['model'] if message.source in config['agents'] else 'default'
            if source == "controller":
                model = config['agents']['planner']['model']
            try:
                
                f.write(f"# {source} ({model}) ({message.type}): \n\n{message.content}\n\n")
                output_list.append({"source": source, "model": model, "type": message.type, "content": message.content})
            except:
                f.write(str(message))
                output_list.append({"source": source, "model": model, "type": None, "content": str(message)})
    
    # Save the output list to a json file
    try:
        output_list_path = os.path.join(output_dir, f"{task.stage}_{task.id}_messages.json")
        with open(output_list_path, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
    except:
        pass

    return output_path

def do_worker(task: Task, loop: asyncio.AbstractEventLoop):
    global_config = get_config(task.bind_config, task.db_path).global_config
    # Run the async function and get the result
    context: ProcessContext = loop.run_until_complete(runners[task.stage](task, global_config))
    
    if task.stage in ["build", "refine", "assemble"]:
        output_dir = os.path.join(os.path.dirname(task.db_path), "machine", task.bind_machine if task.bind_machine else task.id)
    else:
        output_dir = os.path.join(os.path.dirname(task.db_path), task.stage)
    
    os.makedirs(output_dir, exist_ok=True)
    db_path = task.db_path

    if task.stage == "plan":
        output_path = os.path.join(output_dir, f"{task.stage}_{task.id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context.result, f, ensure_ascii=False, indent=2)

    if task.stage != "simulation": # For simulation, the result is a csv file
        output_path = save_context(context = context.context, task = task, output_dir = output_dir)
    else:
        output_path = context.result

    mark_task_completed(task_id=task.id, result_path=output_path, db_path=db_path, token_input=context.token_input, token_output=context.token_output, turn=context.turn)

def worker_process(task: Task):
    print(f"Starting worker for Task {task.id} ({task.stage})")
    
    loop = init_loop()
    
    try:
        do_worker(task, loop)
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        
        print(f"Error in worker process for task {task.id}: {error_type} - {error_msg}")
            
        # Mark task as failed in the database
        from scheduler.task_db import mark_task_failed
        mark_task_failed(task.id, f"{error_type}: {error_msg}", task.db_path)
    finally:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Wait for all tasks to be cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        # Close the loop
        loop.close()

def test_worker():
    parser = argparse.ArgumentParser(description='Test worker with task ID and database path')
    parser.add_argument('--id', type=str, required=True, help='Task ID to process')
    parser.add_argument('--db', type=str, required=True, help='Path to project database file')
    args = parser.parse_args()
    task_id = args.id
    db_path = args.db
    assert os.path.exists(db_path), f"Database file {db_path} does not exist"
    task = get_task(task_id, db_path=db_path)
    loop = init_loop()
    do_worker(task, loop)
    
def check_verify_result(verify_result: str | bool) -> bool:
    if isinstance(verify_result, str):
        if "true" in verify_result.lower():
            approved = True
        else:
            approved = False
    elif isinstance(verify_result, bool):
        approved = verify_result
        
    return approved
    
if __name__ == "__main__":
    test_worker()