import sys, os
import json
import argparse
sys.path.append(os.getcwd())
from ReasonFlux.utils.client import initialize_hierarchical_database

def config()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use this script to create a hierarchical database storage for ReasonFlux reformatted template")
    parser.add_argument("--database_config", type=str, default="ReasonFlux/config/database/database.yaml", help="The configuration file for the hierarchical database")
    parser.add_argument("--template_file", type=str, default="data/format_library.json", help="The file containing the reformatted template")
    parser.add_argument("--overwrite", type=bool, default=False, help="Whether to overwrite the database")
    args = parser.parse_args()
    return args

def main():
    args = config()
    database = initialize_hierarchical_database(args.database_config)
    os.makedirs(database.data_dir, exist_ok=True)

    if args.overwrite:
        print("Clearing the database...")
        database.clear()
        print("Database cleared.")
    
    with open(args.template_file, "r") as f:
        template_data = json.load(f)
        print(f"Load {len(template_data)} templates from {args.template_file}")
    
    print("Adding templates to the database...")
    database.add_recursive_dict(template_data)
    print(f"Database creation completed. Database path: {database.data_dir}")
    for i in range(database.max_level):
        level_name = f"level_{i}"
        print(f"Level {i}: {database.collections[level_name].count()} nodes")

if __name__ == "__main__":
    main()
# python scripts/create_hierarchical_database.py --database_config ReasonFlux/config/database/database.yaml --template_file data/format_library.json