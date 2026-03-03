import json
import sys
from pathlib import Path

def convert_json_to_jsonl(input_path, output_path=None):
    """
    Converts a JSON file (list of objects) to a JSONL file.
    """
    input_file = Path(input_path)
    
    # If no output path is provided, use the same name with .jsonl extension
    if output_path is None:
        output_file = input_file.with_suffix('.jsonl')
    else:
        output_file = Path(output_path)

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure the JSON data is a list
        if not isinstance(data, list):
            print(f"Error: Expected a list of objects in {input_path}, but got {type(data).__name__}.")
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                # Convert each dictionary/object to a string and write to a new line
                json_record = json.dumps(entry, ensure_ascii=False)
                f.write(json_record + '\n')

        print(f"Successfully converted: {input_file.name} -> {output_file.name}")

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON. Check if {input_path} is valid.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Usage: python script.py input.json [output.jsonl]
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_json_file>")
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
        convert_json_to_jsonl(in_path, out_path)