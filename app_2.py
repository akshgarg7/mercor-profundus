import time
import datetime
import json
import random
import urllib
import streamlit as st
import llm_openrouter as llm
import dotenv
import os
import requests
import json
import dotenv 
import os 
import re
# from add_to_airtable import insertion_wrapper, set_to_wip, get_record_id_from_task_id
# from streamlit_extras.stylable_container import stylable_container

dotenv.load_dotenv()

st.set_page_config(page_title="Multi LLM Test Tool", layout="wide")


get_url = 'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask/?maxRecords=1000&view=Grid%20view'
get_headers = {
    'Authorization': 'Bearer ' + os.getenv('AIRTABLE_API_KEY'),
    'Content-Type': 'application/json'
}
# Fetch all records
def get_record_id_from_task_id(target_task_id):
    # st.markdown(get_headers)
    response = requests.get(get_url, headers=get_headers)
    data = response.json()
    # st.markdown(data)
    records = data.get('records', [])
    # st.header(records)
    # print(records)
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
    response = requests.patch(update_url, headers=get_headers, data=json.dumps(data))
    # print(response.text)

# Design the data
def prepare_data(prompt, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    return {
        "fields": {
            "Prompt": prompt,
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
    # print("Headers:", get_headers)
    # print("Update URL:", update_url)
    # print("Data:", json.dumps(data))
    response = requests.patch(update_url, headers=get_headers, data=json.dumps(data))
    # print(response.text)

def insertion_wrapper(record_id, prompt, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    # record_id = get_record_id_from_task_id(task_id)
    patch_data = prepare_data(prompt, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response)
    update_matching_record(record_id, patch_data)


specific_model_ids = [
    'anthropic/claude-3.5-sonnet',
    'google/gemini-pro-1.5',
    'openai/gpt-4o',
    # Add other model IDs as needed
]

gemini_models = [
    # 'google/gemini-flash-1.5',
    'google/gemini-pro-1.5',
]

@st.cache_data
def get_models():
    models = llm.available_models()
    models = [model for model in models if model.id in specific_model_ids]
    models = sorted(models, key=lambda x: x.name)
    return models


def prepare_session_state():
    # print(f"AIRTABLE API KEY: {os.getenv('AIRTABLE_API_KEY')}")
    # st.header(f"AIRTABLE API KEY: {os.getenv('AIRTABLE_API_KEY')}")
    models = get_models()
    gemini_models_parsed = [model for model in models if model.id in gemini_models]
    first_model = random.choice(gemini_models_parsed)
    remaining_models = [model for model in models if model.id != first_model.id]
    second_model = random.choice(remaining_models)
    picked_models = set([first_model, second_model])
    remaining_models = set(models) - picked_models
    third_model = random.choice(list(remaining_models))

    # randomly permute [first_model, second_model]
    first_model, second_model = random.sample([first_model, second_model], 2)
    print(first_model, second_model, third_model)
    session_vars = {  # Default values
        "prompt": "",
        "temperature": 0.0,
        "max_tokens": 2048,
        "models": [first_model, second_model, third_model],  # Select two models randomly
        "third_model": third_model,
        "response": {},
        "cost_and_stats": {},
    }

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = session_vars[var]

    print(st.session_state['models'])
    print(f"first_model: {first_model}: {st.session_state['models'][0]}")


def configuration():
    models = st.session_state.models
    st.session_state.prompt = ""
    st.session_state.temperature = 0
    st.session_state.max_tokens = 2048


def get_llm_response(user_input: str) -> dict[llm.Model, llm.LLMResponse]:
    with st.spinner("Sending request..."):
        models = st.session_state.models
        if not isinstance(models, list):
            models = [models]  # Ensure models is a list even if only one model is selected
        if not models:
            st.error("No models selected. Please select at least one model.")
            st.stop()

        # print(f"models: {models}")
        # print(f"prompt: {st.session_state.prompt}")
        # print(f"user_input: {user_input}")
        # print(f"temperature: {st.session_state.temperature}")
        # print(f"max_tokens: {st.session_state.max_tokens}")
        response = llm.chat_completion_multiple(
            models, st.session_state.prompt, user_input, st.session_state.temperature, st.session_state.max_tokens
        )
    return response


def get_cost_and_stats(response: dict[llm.Model, llm.LLMResponse]) -> dict[llm.Model, llm.LLMCostAndStats]:
    with st.spinner("Calculating cost and stats..."):
        cost_and_stats = llm.cost_and_stats_multiple(response)
    return cost_and_stats

def rewrite_markdown(input_text):
    # Regular expression to match code blocks with a language tag
    pattern = re.compile(r'```(\w+)\n(.*?)\n```', re.DOTALL)
    
    # Function to replace matched code blocks with regular markdown
    def replace_code_block(match):
        code = match.group(2)  # Extract the code content
        return f'```\n{code}\n```'
    
    # Substitute all matching code blocks
    result = pattern.sub(replace_code_block, input_text)
    
    return result

def show_response(response: dict[llm.Model, llm.LLMResponse], cost_and_stats: dict[llm.Model, llm.LLMCostAndStats]):
    # Sort the response by model name to keep the order consistent
    # response = dict(sorted(response.items(), key=lambda x: x[0].name))
    orig_model_order = [model for model in st.session_state.models]
    # print(orig_model_order)
    response = {model: response[model] for model in orig_model_order}
    response = [(model, response[model]) for model in orig_model_order]

    # Show the response side by side by model
    cols = st.columns(2)
    for i, (m, r) in enumerate(response):
        if i >= 2: 
            continue
        # print(r)
        print('-----------------------------------')
        with cols[i]:
            st.markdown(f"### Model {m.name}")

#             print(f"""
# <div style="border:2px solid #ccc; border-radius: 5px; padding: 10px; margin-top: 5px;">
# {r.response.strip()}
# </div>
#             """.strip())

            # print(rewrite_markdown(r.response.strip()))
            st.markdown(f"""
{rewrite_markdown(r.response.strip())}
            """.strip(), unsafe_allow_html=True) 
            if 'model_outputs' not in st.session_state:
                st.session_state['model_outputs'] = []
            st.session_state['model_outputs'].append(rewrite_markdown(r.response.strip()))

    st.header("")

    model_a_name = st.session_state.models[0].name
    model_b_name = st.session_state.models[1].name
    model_c_name = st.session_state.models[2].name
    model_a_response = st.session_state.response[st.session_state.models[0]].response
    model_b_response = st.session_state.response[st.session_state.models[1]].response
    model_c_response = st.session_state.response[st.session_state.models[2]].response
    log_evaluation_data(model_a_name, model_b_name, model_c_name, model_a_response, model_b_response, model_c_response)
    # cols = st.columns(3)
    # with cols[1]:
    # with stylable_container(
    #     "green",
    #     css_styles="""
    #     button {
    #         background-color: #6366f1;
    #         color: white;
    #     }
    #     """
    # ):
    if st.button("Submit Evaluation (This will log your answers to airtable, please make sure at least one model is failing)", type='primary'):
        model_c_response = llm.chat_completion_multiple(
            [st.session_state.third_model], st.session_state.prompt, user_input, st.session_state.temperature, st.session_state.max_tokens
        )
        # model_c_response = model_c_response[st.session_state.third_model].response
        # model_c_response = rewrite_markdown(model_c_response.strip())
        model_a_name = st.session_state.models[0].name
        model_b_name = st.session_state.models[1].name
        model_c_name = st.session_state.models[2].name
        model_a_response = st.session_state.response[st.session_state.models[0]].response
        model_b_response = st.session_state.response[st.session_state.models[1]].response
        model_c_response = st.session_state.response[st.session_state.models[2]].response
        log_evaluation_data(model_a_name, model_b_name, model_c_name, model_a_response, model_b_response, model_c_response)

        # print(f'model_a_name: {model_a_name}')
        # print(f'model_b_name: {model_b_name}')
        # print(f'model_c_name: {model_c_name}')
        # print(f'model_a_response: {model_a_response}')
        # print(f'model_b_response: {model_b_response}')
        # print(f'model_c_response: {model_c_response}')
        


def show_evaluation_form():
    st.header("Evaluate the Responses")
    col1, col2 = st.columns(2)

    # Define the labels
    labels = {
        1: 'Really Bad',
        2: 'Bad',
        3: 'Average',
        4: 'Good',
        5: 'Excellent'
    }

    label_style = "font-size:20px;" 

    with col1:
        st.subheader("Model 1 Evaluation")
        st.markdown(f"<div style='{label_style}'><b>Accuracy and Correctness of Code </b><br> Measures whether the response is technically correct and functions as expected. </div>", unsafe_allow_html=True)
        accuracy1 = st.slider("", 1, 5, key="accuracy1")
        accuracy1_confirm = st.checkbox("Confirm Accuracy", key="accuracy1_confirm")
        st.markdown(f"<div style='{label_style}'><b>Relevance and Completeness  </b><br> Assesses whether the response is relevant to the prompt and fully addresses the task requirements.</div>", unsafe_allow_html=True)
        relevance1 = st.slider(" ", 1, 5, key="relevance1")
        relevance1_confirm = st.checkbox("Confirm Relevance", key="relevance1_confirm")
        st.markdown(f"<div style='{label_style}'><b>Readability and Documentation </b><br> Assesses whether the response is easy to understand and maintain.</div>", unsafe_allow_html=True)
        conciseness1 = st.slider("  ", 1, 5, key="conciseness1")
        conciseness1_confirm = st.checkbox("Confirm Conciseness", key="conciseness1_confirm")

    with col2:
        st.subheader("Model 2 Evaluation")

        st.markdown(f"<div style='{label_style}'><b>Accuracy and Correctness of Code </b><br> Measures whether the response is technically correct and functions as expected. </div>", unsafe_allow_html=True)
        accuracy2 = st.slider("   ", 1, 5, key="accuracy2")
        accuracy2_confirm = st.checkbox("Confirm Accuracy", key="accuracy2_confirm")
        st.markdown(f"<div style='{label_style}'><b>Relevance and Completeness </b><br> Assesses whether the response is relevant to the prompt and fully addresses the task requirements.</div>", unsafe_allow_html=True)
        relevance2 = st.slider("    ", 1, 5, key="relevance2")
        relevance2_confirm = st.checkbox("Confirm Relevance", key="relevance2_confirm")
        st.markdown(f"<div style='{label_style}'><b>Readability and Documentation </b><br> Assesses whether the response is easy to understand and maintain.</div>", unsafe_allow_html=True)
        conciseness2 = st.slider("     ", 1, 5, key="conciseness2")
        conciseness2_confirm = st.checkbox("Confirm Conciseness", key="conciseness2_confirm")

    # Increase the font size for the "Which model do you prefer?" question using CSS
    st.markdown("""
    <style>
        div[data-testid="stRadio"] > label > div {
            font-size: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Which model do you prefer?")
    preferred_model = st.radio(
        "",
        ('Model 1', 'Model 2', "Equal (Both Good)", "Equal (Both Bad)")
    )

    st.subheader("Reason for preference")
    reason_for_preference = st.text_area("")

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("Submit Evaluation (You will not be able to go back and modify your answers)"):
        if reason_for_preference == "" or not (accuracy1_confirm and relevance1_confirm and conciseness1_confirm and accuracy2_confirm and relevance2_confirm and conciseness2_confirm):
            st.error("Please confirm all selections and provide a reason for your preference before submitting.")
        else:
            st.write("You submitted:")
            st.write(f"Model 1 - Accuracy: {labels[accuracy1]}, Relevance: {labels[relevance1]}, Conciseness: {labels[conciseness1]}")
            st.write(f"Model 2 - Accuracy: {labels[accuracy2]}, Relevance: {labels[relevance2]}, Conciseness: {labels[conciseness2]}")
            st.write(f"Preferred Model: {preferred_model}")
            st.write(f"Reason for preference: {reason_for_preference}")

            # Here you can log the data to a file or database
            log_evaluation_data(st.session_state.models[0].name, st.session_state.models[1].name, accuracy1, relevance1, conciseness1, accuracy2, relevance2, conciseness2, preferred_model, reason_for_preference)

            st.session_state['prompt'] = ""
            st.session_state['temperature'] = 0.0
            st.session_state['max_tokens'] = 2048
            # st.session_state['models'] = random.sample(get_models(), 2)  # Reselect two random models
            st.session_state['response'] = {}
            st.session_state['cost_and_stats'] = {}
            st.session_state['model_outputs'] = []
            
            # Display a success message
            st.success("Evaluation submitted successfully. Fields will reset in 5 seconds.")
            time.sleep(5)
            st.experimental_rerun()

import pandas as pd
import os

def log_evaluation_data(model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    insertion_wrapper(st.session_state['record_id'], user_input, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response)

# def log_evaluation_data(email, model_left, model_right, accuracy1, relevance1, conciseness1, accuracy2, relevance2, conciseness2, preferred_model, reason_for_preference):
#     # Define the path for the CSV file
#     csv_file_path = 'evaluation_data.csv'
    
#     # Create a DataFrame from the input data
#     data = {
#         "Email": [email],
#         "Model Left": [model_left],
#         "Model Right": [model_right],
#         "Model 1 Accuracy": [accuracy1],
#         "Model 1 Relevance": [relevance1],
#         "Model 1 Conciseness": [conciseness1],
#         "Model 2 Accuracy": [accuracy2],
#         "Model 2 Relevance": [relevance2],
#         "Model 2 Conciseness": [conciseness2],
#         "Preferred Model": [preferred_model],
#         "Reason for Preference": [reason_for_preference],
#         "Model 1 Output": [st.session_state['model_outputs'][0]],  # Store Model 1 output as a JSON string
#         "Model 2 Output": [st.session_state['model_outputs'][1]],
#         "status": ["submitted"],
#         'query': [task_id]
#     }

#     insertion_wrapper(st.session_state['record_id'], email, model_left, model_right, accuracy1, relevance1, conciseness1, accuracy2, relevance2, conciseness2, preferred_model, st.session_state['model_outputs'][0], st.session_state['model_outputs'][1])
#     df = pd.DataFrame(data)
    
#     # Check if the file exists to decide whether to write headers
#     if os.path.isfile(csv_file_path):
#         # File exists, append without writing the header
#         df.to_csv(csv_file_path, mode='a', header=False, index=False)
#     else:
#         # File does not exist, write with the header
#         df.to_csv(csv_file_path, mode='w', header=True, index=False)

# Parse URL parameters
query_params = st.experimental_get_query_params()

# Assuming there's a 'query' parameter in the URL
task_id = query_params.get("query", [""])[0]  # Get 'query' parameter, default to empty string if not present
task_type = query_params.get("task_type", [""])[0] 
language=query_params.get("language", [""])[0]
using_framework=query_params.get("using_framework", [""])[0]
frameworks_used = query_params.get("frameworks_used", [""])[0]
record_id = query_params.get("record_id", [""])[0]
st.session_state['record_id'] = record_id


# task_id = 1

st.title(f"Mercor Data Collection Pilot")
st.markdown(f"**Task ID:** {task_id}")
st.markdown(f"**Task Type:** {task_type}")
st.markdown(f"**Language:** {language}")
st.markdown(f"**Using Framework:** {using_framework}")
st.markdown(f"**Frameworks Used:** {frameworks_used}")
st.markdown("---")  # Add a horizontal line for better separation



# At the top of your Streamlit app, after setting page configurations
# email = st.text_input("Enter your email so we can compensate you fairly", "")


# Use the default_query as the default value in a text input
# display the query text
# st.write(f"Query: {default_query}")

# user_input = st.text_input("Enter your query:", value=default_query)

prepare_session_state()

configuration()
st.markdown("""
<style>
.expandable-textarea {
    width: 100%;
    min-height: 100px;
    padding: 10px;
    font-size: 16px;
    border: 1px solid #ccc;
    border-radius: 5px;
    resize: vertical;
}
</style>
""", unsafe_allow_html=True)

user_input = st.text_area("Input Prompt", placeholder="Enter the prompt here", height=400, key="expandable_input", help="Type your prompt here. The text area will expand as you type.")
# user_input = st.text_area("Input Prompt", placeholder="Enter the prompt here", height=100)
read_and_agreed = True
# send_button = st.button("Send Request")

st.markdown("<br>", unsafe_allow_html=True)  # Add space above the button
# with stylable_container(
#         "green",
#         css_styles="""
#         button {
#             background-color: #6366f1;
#             color: white;
#         }
#         """
#     ):
send_button = st.button("Send Request", key="send_button", help="Click to send your request", type='primary')

# Add custom CSS for the button
st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    font-size: 18px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

def get_filled_data(record_id):
    retrieval_url = f'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask/{record_id}'
    response = requests.get(retrieval_url, headers=get_headers)
    data = response.json()
    prev_data = data['']
    print(data)
    # return data['fields']['']
if send_button:
    get_filled_data(st.session_state['record_id'])

    # st.session_state['record_id'] = get_record_id_from_task_id(int(task_id))
    # st.markdown(f"record_id: {st.session_state['record_id']}")
    # set_to_wip(st.session_state['record_id'])
    if not read_and_agreed:
        st.error("Please confirm that there is no private or sensitive information in your request")
        st.stop()
    if not st.session_state.models:
        st.error("Please select at least one model")
        st.stop()
    if not user_input:
        st.error("Please enter a request")
        st.stop()

    st.session_state.response = get_llm_response(user_input)
    # st.session_state['model1_response'] = st.session_state.response[st.session_state.models[0]]
    # st.session_state['model2_response'] = st.session_state.response[st.session_state.models[1]]
    # st.session_state.cost_and_stats = get_cost_and_stats(st.session_state.response)

if st.session_state.response:
    show_response(st.session_state.response, st.session_state.cost_and_stats)

    # Let the users download the results in JSON format
    response_json = {}
    for model, response in st.session_state.response.items():
        response_json[model.name] = {
            "response": response.to_dict(),
            # "cost_and_stats": st.session_state.cost_and_stats[model].to_dict(),
        }
    response_json = json.dumps(response_json, indent=4)

    # show_evaluation_form()

