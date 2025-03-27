import json

def transform_data(data):
    """
    Recursively process nested dictionaries to transform the last layer of lists.
    Each dictionary in the list is converted into a key-value pair where the key is 'template_name' and the value is the JSON string representation of the dictionary.
    """
    transformed_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            transformed_data[key] = transform_data(value)
        elif isinstance(value, list):
            transformed_list = {}
            for item in value:
                if isinstance(item, dict) and "template_name" in item:
                    template_name = item["template_name"]
                    transformed_list[template_name] = json.dumps(item, ensure_ascii=False)
            transformed_data[key] = transformed_list
        else:
            transformed_data[key] = value
    return transformed_data

input_file_path = "data/template_library.json"
output_file_path = "data/format_library.json"

try:
    with open(input_file_path, "r", encoding="utf-8") as infile:
        original_data = json.load(infile)
    
    transformed_data = transform_data(original_data)
    
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump(transformed_data, outfile, indent=4, ensure_ascii=False)
    
    print(f"Data has been successfully formatted and saved to {output_file_path}")
except FileNotFoundError:
    print(f"Error: The file {input_file_path} was not found. Please ensure the file path is correct.")
except json.JSONDecodeError:
    print(f"Error: The file {input_file_path} is not a valid JSON format.")
except Exception as e:
    print(f"An error occurred: {e}")