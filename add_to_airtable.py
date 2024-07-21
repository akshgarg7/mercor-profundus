import requests
import json
import dotenv 
import os 

dotenv.load_dotenv()

# API URL and headers
url = 'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask?maxRecords=3&view=Grid%20view'
headers = {
    'Authorization': 'Bearer ' + os.getenv('AIRTABLE_API_KEY'),
    'Content-Type': 'application/json'
}
# Fetch all records
def get_record_id_from_task_id(target_task_id):
    response = requests.get(url, headers=headers)
    data = response.json()
    records = data.get('records', [])
    for record in records:
        task_id = record['fields']['Subtask Id']
        if task_id == target_task_id:
            return record['id']
    return None

def set_to_wip(record_id):
    data = {
        "fields": {
            "status": "in_progress"
        }
    }
    update_url = f'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask/{record_id}'
    response = requests.patch(update_url, headers=headers, data=json.dumps(data))
    print(response.text)

# Design the data
def prepare_data(model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    return {
        "fields": {
            "Model A Model Name": model_a,
            "Model B Model Name": model_b,
            "Model C Model Name": model_c,
            "Model A Response": model_a_response,
            "Model B Response": model_b_response,
            "Model C Response": model_c_response,
        }
    }

# 
def update_matching_record(record_id, data):
    update_url = f'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask/{record_id}'
    response = requests.patch(update_url, headers=headers, data=json.dumps(data))
    print(response.text)

def insertion_wrapper(record_id, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    # record_id = get_record_id_from_task_id(task_id)
    patch_data = prepare_data(model_a, model_b, model_c, model_a_response, model_b_response, model_c_response)
    update_matching_record(record_id, patch_data)

if __name__ == "__main__":

    record_id = get_record_id_from_task_id(188)
    insertion_wrapper(record_id, 'gemini', 'gpt-4o', 'claude', 'output a that i just generated', 'output b', 'output c')

    # set_to_wip(record_id)


