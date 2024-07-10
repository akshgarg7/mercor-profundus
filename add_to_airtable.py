import requests
import json

# API URL and headers
url = 'https://api.airtable.com/v0/appCRqdHpEGeE1cE7/Tasks'
headers = {
    'Authorization': 'Bearer pat90qSxYJL8DcBQY.81e2907b757aa5692345f323d85c9c711f57ea5b0560d15700339708a4a983c9',
    'Content-Type': 'application/json'
}

# Data to be posted
data = {
    "records": [
        {
            "fields": {
                "assigned_to": "akshgarg@stanford.edu",
                "status": "unstarted",
                "task_type": "Generating",
                "language": "JavaScript",
                "is_using_framework_or_library": "true",
                "num_turns": "single_turn",
                "context_length": "Long"
            }
        }
    ]
}

# # Make the POST request
# response = requests.post(url, headers=headers, data=json.dumps(data))



# # Print the response from the server
# print(response.text)


# list_records_url = 'https://api.airtable.com/v0/appCRqdHpEGeE1cE7/Tasks'
# response = requests.get(list_records_url, headers=headers)
# print(response.text)

patch_data = {
   "fields": {
        "assigned_to": "akshgarg@stanford.edu",
        "status": "unstarted",
        "task_type": "Generating",
        "language": "JavaScript",
        "is_using_framework_or_library": "true",
        "num_turns": "single_turn",
        "context_length": "Long"
    }
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

def update_matching_record(record_id, data):
    update_url = f'https://api.airtable.com/v0/appCRqdHpEGeE1cE7/Tasks/{record_id}'
    response = requests.patch(update_url, headers=headers, data=json.dumps(data))
    print(response.text)

record_id = get_record_id_from_task_id(5)
update_matching_record(record_id, patch_data)


record_id = get_record_id_from_task_id(6)
patch_data = prepare_data('akshgarg@gmail.com', 'gpt-4o', 'gpt-4o', 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 'Model 1', 'output', 'output')
update_matching_record(record_id, patch_data)