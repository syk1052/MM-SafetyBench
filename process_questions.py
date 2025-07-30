import json
import os

def process_questions_json(input_file_path, output_file_path, model_responses=None):
    """
    Process the questions JSON file to add model responses.
    
    Args:
        input_file_path (str): Path to input JSON file
        output_file_path (str): Path to output JSON file
        model_responses (dict): Dictionary with question_id as key and model responses as value
    """
    
    # Read the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each question
    for question_id, question_data in data.items():
        # Initialize the ans field
        question_data["ans"] = {}
        
        # Add LLaVA-1.5(7B) model response
        if model_responses and question_id in model_responses:
            # If we have actual responses, use them
            question_data["ans"]["LLaVA-1.5(7B)"] = {
                "text": model_responses[question_id].get("LLaVA-1.5(7B)", "")
            }
        else:
            # Placeholder for model response
            question_data["ans"]["LLaVA-1.5(7B)"] = {
                "text": ""  # Empty string for now, to be filled with actual response
            }
    
    # Write the output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(data)} questions and saved to {output_file_path}")

def add_model_responses(input_file_path, output_file_path, model_name, responses_dict):
    """
    Add responses from a specific model to the JSON file.
    
    Args:
        input_file_path (str): Path to input JSON file
        output_file_path (str): Path to output JSON file  
        model_name (str): Name of the model
        responses_dict (dict): Dictionary with question_id as key and response text as value
    """
    
    # Read the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add responses for the specified model
    for question_id, response_text in responses_dict.items():
        if question_id in data:
            # Initialize ans field if it doesn't exist
            if "ans" not in data[question_id]:
                data[question_id]["ans"] = {}
            
            # Add the model response
            data[question_id]["ans"][model_name] = {
                "text": response_text
            }
        else:
            print(f"Warning: Question ID {question_id} not found in data")
    
    # Write the output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Added {model_name} responses and saved to {output_file_path}")

def main():
    # File paths
    input_file = "../data/MM-SafetyBench/processed_questions/01-Illegal_Activitiy.json"
    output_file = "../data/MM-SafetyBench/answer_format/01-Illegal_Activitiy_with_responses.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the questions and add the structure for model responses
    process_questions_json(input_file, output_file)
    
    # Example of how to add actual model responses
    # You can uncomment and modify this when you have actual responses
    """
    example_responses = {
        "0": "I cannot and will not provide advice on illegal activities such as hacking into bank accounts...",
        "1": "I cannot provide information on purchasing illegal drugs...",
        # Add more responses as needed
    }
    
    add_model_responses(
        output_file, 
        output_file, 
        "LLaVA-1.5(7B)", 
        example_responses
    )
    """

if __name__ == "__main__":
    main() 