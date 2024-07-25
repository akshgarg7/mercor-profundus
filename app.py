import json
import random
import streamlit as st
import llm_openrouter as llm
import dotenv
import os
import requests
import re
import time
import concurrent.futures

dotenv.load_dotenv()

st.set_page_config(page_title="Multi LLM Test Tool", layout="wide")

get_url = 'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask/?maxRecords=1000&view=Grid%20view'
get_headers = {
    'Authorization': 'Bearer ' + os.getenv('AIRTABLE_API_KEY'),
    'Content-Type': 'application/json'
}

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

def update_matching_record(record_id, data):
    update_url = f'https://api.airtable.com/v0/appv7hUQouIL8ckzC/Writing%20Subtask/{record_id}'
    max_retries = 5
    for attempt in range(max_retries):  # New code starts here
        try:
            response = requests.patch(update_url, headers=get_headers, data=json.dumps(data))
            response.raise_for_status()
            break  # exit the retry loop if the request is successful
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                st.warning(f"Airtable is experiencing some slowdown. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error("Failed to update Airtable after multiple attempts. Please try again later.")
                raise e  # New code ends here

def insertion_wrapper(record_id, prompt, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    futures = []
    patch_data = prepare_data(prompt, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response)
    update_matching_record(record_id, patch_data)

specific_model_ids = [
    'anthropic/claude-3.5-sonnet',
    'google/gemini-pro-1.5',
    'openai/gpt-4o',
]

gemini_models = [
    'google/gemini-pro-1.5',
]

@st.cache_data
def get_models():
    models = llm.available_models()
    models = [model for model in models if model.id in specific_model_ids]
    models = sorted(models, key=lambda x: x.name)
    return models

def prepare_session_state():
    models = get_models()
    gemini_models_parsed = [model for model in models if model.id in gemini_models]
    first_model = random.choice(gemini_models_parsed)
    remaining_models = [model for model in models if model.id != first_model.id]
    second_model = random.choice(remaining_models)
    picked_models = set([first_model, second_model])
    remaining_models = set(models) - picked_models
    third_model = random.choice(list(remaining_models))

    first_model, second_model = random.sample([first_model, second_model], 2)
    session_vars = {
        "prompt": "",
        "temperature": 0.0,
        "max_tokens": 100000,
        "models": [first_model, second_model, third_model],
        "third_model": third_model,
        "response": {},
        "cost_and_stats": {},
    }

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = session_vars[var]
            if var == "models":
                print(f"Selected models: {', '.join([model.name for model in st.session_state.models])}")


def get_llm_response(user_input: str) -> dict[llm.Model, llm.LLMResponse]:
    with st.spinner("Sending request..."):
        models = st.session_state.models
        if not isinstance(models, list):
            models = [models]
        if not models:
            st.error("No models selected. Please select at least one model.")
            st.stop()

        response = llm.chat_completion_multiple(
            models, st.session_state.prompt, user_input, st.session_state.temperature, st.session_state.max_tokens
        )
    return response

def rewrite_markdown(input_text):
    pattern = re.compile(r'```(\w+)\n(.*?)\n```', re.DOTALL)
    def replace_code_block(match):
        code = match.group(2)
        return f'```\n{code}\n```'
    result = pattern.sub(replace_code_block, input_text)
    return result

def show_response(response: dict[llm.Model, llm.LLMResponse]):
    orig_model_order = [model for model in st.session_state.models]
    response = {model: response[model] for model in orig_model_order}
    response = [(model, response[model]) for model in orig_model_order]

    cols = st.columns(2)
    for i, (m, r) in enumerate(response):
        if i >= 2: 
            continue
        with cols[i]:
            st.markdown(f"### Model {i+1}")
            st.markdown(f"""
{rewrite_markdown(r.response.strip())}
            """.strip(), unsafe_allow_html=True)
            if 'model_outputs' not in st.session_state:
                st.session_state['model_outputs'] = []
            st.session_state['model_outputs'].append(rewrite_markdown(r.response.strip()))

    st.header("")

    print(st.session_state.models[0], st.session_state.models[1], st.session_state.models[2])
    model_a_name = st.session_state.models[0].name
    model_b_name = st.session_state.models[1].name
    model_c_name = st.session_state.models[2].name
    model_a_response = st.session_state.response[st.session_state.models[0]].response
    model_b_response = st.session_state.response[st.session_state.models[1]].response
    model_c_response = st.session_state.response[st.session_state.models[2]].response
    log_evaluation_data(model_a_name, model_b_name, model_c_name, model_a_response, model_b_response, model_c_response)

def log_evaluation_data(model_a, model_b, model_c, model_a_response, model_b_response, model_c_response):
    insertion_wrapper(st.session_state['record_id'], user_input, model_a, model_b, model_c, model_a_response, model_b_response, model_c_response)

query_params = st.experimental_get_query_params()
task_id = query_params.get("query", [""])[0]
task_type = query_params.get("task_type", [""])[0] 
language = query_params.get("language", [""])[0]
using_framework = query_params.get("using_framework", [""])[0]
frameworks_used = query_params.get("frameworks_used", [""])[0]
record_id = query_params.get("record_id", [""])[0]
st.session_state['record_id'] = record_id

st.title(f"Mercor Data Collection Pilot")
st.markdown(f"**Task ID:** {task_id}")
st.markdown(f"**Task Type:** {task_type}")
st.markdown(f"**Language:** {language}")
st.markdown(f"**Using Framework:** {using_framework}")
st.markdown(f"**Frameworks Used:** {frameworks_used}")
st.markdown("---")

prepare_session_state()
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
read_and_agreed = True

st.markdown("<br>", unsafe_allow_html=True)
send_button = st.button("Send Request", key="send_button", help="Click to send your request", type='primary')

if send_button:
    st.session_state['send_button_clicked'] = True

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
    max_retries = 5
    for attempt in range(max_retries):  # New code starts here
        try:
            response = requests.get(retrieval_url, headers=get_headers)
            response.raise_for_status()
            data = response.json()
            saved_prompt = data['fields'].get('Prompt', "")
            submitted = data['fields'].get('Status', "")
            return saved_prompt == "", submitted == "Done"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"Airtable is experiencing some slowdown. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.error("Failed to retrieve data from Airtable after multiple attempts. Please try again later.")
                raise e  # New code ends here
            
checkbox = st.checkbox("Overwrite Prompt", value=False)
ok_to_proceed, already_submitted = get_filled_data(st.session_state['record_id'])


if send_button:
    if already_submitted:
        st.error("You can not modify an already submitted response. Contact Sid for assistance")
        st.stop() 

    if not user_input:
        st.session_state['send_button_clicked'] = False
        st.session_state['response'] = None
        st.error("Please enter a request")
        st.stop()

    if not ok_to_proceed and not checkbox:
        st.session_state['send_button_clicked'] = False
        st.session_state['response'] = None
        st.warning("Please confirm that you want to overwrite the prompt and submit again")
        st.stop()

    st.session_state.response = get_llm_response(user_input)
    st.session_state.send_button_clicked = False

if st.session_state.response:
    show_response(st.session_state.response)

    response_json = {}
    for model, response in st.session_state.response.items():
        response_json[model.name] = {
            "response": response.to_dict(),
        }
    response_json = json.dumps(response_json, indent=4)

