import sys
import json

def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python dictionary.

    Parameters:
        - file_path: A string representing the path to the JSON file.

    Returns:
        A dictionary containing the data from the JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
