import requests
import json
import dotenv 
import os 

dotenv.load_dotenv()

# API URL and headers
url = 'https://api.airtable.com/v0/appCRqdHpEGeE1cE7/Tasks'
headers = {
    'Authorization': 'Bearer ' + os.getenv('AIRTABLE_API_KEY'),
    'Content-Type': 'application/json'
}
# Fetch all records
def get_record_id_from_task_id(task_id):
    response = requests.get(url, headers=headers)
    data = response.json()

    target_task_id = task_id  # Assuming task_id is stored as a string
    record_id = None
    for record in data.get('records', []):
        if record['fields'].get('task_id') == target_task_id:
            record_id = record['id']
            return record_id

    if record_id:
        print(f"Record ID: {record_id}")
    else:
        print("No record found")

# Design the data
def prepare_data(labeler_email, model_left, model_right, model_1_accuracy_score, model_1_relevance_score, model_1_readability_score, model_2_accuracy_score, model_2_relevance_score, model_2_readability_score, preferred_model, model_1_output, model_2_output):
    return {
        "fields": {
            "labeler_email": labeler_email,
            "model_left": model_left,
            "model_right": model_right,
            "model_1_accuracy_score": model_1_accuracy_score,
            "model_1_relevance_score": model_1_relevance_score,
            "model_1_readability_score": model_1_readability_score,
            "model_2_accuracy_score": model_2_accuracy_score,
            "model_2_relevance_score": model_2_relevance_score,
            "model_2_readability_score": model_2_readability_score,
            "preferred_model": preferred_model,
            "model_1_output": model_1_output,
            "model_2_output": model_2_output
        }
    }

# 
def update_matching_record(record_id, data):
    update_url = f'https://api.airtable.com/v0/appCRqdHpEGeE1cE7/Tasks/{record_id}'
    response = requests.patch(update_url, headers=headers, data=json.dumps(data))
    print(response.text)

def insertion_wrapper(task_id, labeler_email, model_left, model_right, model_1_accuracy_score, model_1_relevance_score, model_1_readability_score, model_2_accuracy_score, model_2_relevance_score, model_2_readability_score, preferred_model, model_1_output, model_2_output ):
    record_id = get_record_id_from_task_id(task_id)
    patch_data = prepare_data(labeler_email, model_left, model_right, model_1_accuracy_score, model_1_relevance_score, model_1_readability_score, model_2_accuracy_score, model_2_relevance_score, model_2_readability_score, preferred_model, model_1_output, model_2_output)
    update_matching_record(record_id, patch_data)

if __name__ == "__main__":
    insertion_wrapper(11, 'akshgarg@gmail.com', 'gpt-4o', 'gpt-4o', 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 'Model 1', 'output', 'output')