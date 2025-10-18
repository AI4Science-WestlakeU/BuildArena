"""
Example Usage:
uv run script/run_construction.py --model gpt-4o --category transport --level soft --n_sample 64 --n_worker 4
"""

import argparse
import os
import yaml
from datetime import datetime

from scheduler.task_db import init_db, insert_config, load_config
from scheduler.scheduler import Scheduler
from agents import model_clients

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

def main():
    parser = argparse.ArgumentParser(description='Run project tasks')
    parser.add_argument('--model', '-md', 
                      type=str,
                      required=True,
                      help='Model to use (must be in supported model clients)')
    parser.add_argument('--category', '-c', 
                      type=str,
                      required=True,
                      help='Task category from levels.yaml (e.g., transport, support, lift)')
    parser.add_argument('--level', '-l', 
                      type=str,
                      required=True,
                      help='Difficulty level from levels.yaml (e.g., soft, medium, hard)')
    parser.add_argument('--n_sample', '-ns', 
                      type=int,
                      default=64,
                      required=False,
                      help='Number of samples to generate')
    parser.add_argument('--n_worker', '-nw', 
                      type=int,
                      default=4,
                      required=False,
                      help='Number of workers to use')
    
    # Parse arguments first to check for level-based mode
    args = parser.parse_args()
    
    # Load levels data for validation and help
    levels = load_levels_yaml()
    available_categories, available_levels = get_available_categories_and_levels(levels)
    
    if args.category not in available_categories:
        raise ValueError(f"Category '{args.category}' not found. Available categories: {available_categories}")
    if args.level not in available_levels:
        raise ValueError(f"Level '{args.level}' not found. Available levels: {available_levels}")
    
    # Find the specific task
    task_content = find_level_task(levels, args.category, args.level)
    if not task_content:
        raise ValueError(f"No task found for category '{args.category}' and level '{args.level}'")
    
    # Validate model if provided
    if args.model:
        if args.model not in model_clients.keys():
            available_models = list(model_clients.keys())
            raise ValueError(f"Model '{args.model}' is not supported. Available models: {available_models}")
    
    config_path = "prompt.yaml"
    assert args.model, "Model must be specified when using default prompt"

    # Load project configuration
    config = load_config(config_path)
    config['project'] = {}
    global_config = {}

    project_name = f"{args.category}_{args.level}_{args.model.replace(' ', '-').replace('.', '-')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    config['project']['name'] = project_name
    config['project']['goal'] = task_content
    global_config['name'] = project_name
    global_config['goal'] = task_content
    global_config['n_sample'] = args.n_sample
    global_config['max_workers'] = args.n_worker
    
    print(f"Overriding models in config with {args.model}")
    if 'agents' in config:
        for agent_name, agent_config in config['agents'].items():
            if 'model' in agent_config:
                agent_config['model'] = args.model
    
    # Initialize project-specific database
    project_name_for_db = f"{args.category}_{args.level}_{args.model.replace(' ', '-').replace('.', '-')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    db_path = f"./datacache/{project_name_for_db}/task_database.db"
    init_db(db_path)
    
    _ = insert_config(config=config, global_config=global_config, bind_task="root", db_path=db_path)

    # Initialize scheduler with project settings
    scheduler = Scheduler(config=global_config, db_path=db_path)
    scheduler.run(break_on_complete = True)

if __name__ == "__main__":
    main()
