import os
import yaml
from datetime import date
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Load prompts from YAML file
def load_prompts():
    try:
        with open('prompts.yaml', 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise RuntimeError(f"Failed to load prompts.yaml: {e}")

PROMPTS = load_prompts()

# ASI:One API config
ASI_API_KEY = os.getenv("ASI_API_KEY")
ASI_URL = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-mini"

def call_asi_one(prompt: str) -> str:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {ASI_API_KEY}'
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }
    response = requests.post(ASI_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

# Agent Definitions
def data_enrichment_agent(inputs):
    prompt = PROMPTS['enrichment_task_desc'].format(**inputs)
    return call_asi_one(prompt)

def needs_analysis_agent(inputs):
    prompt = PROMPTS['needs_analysis_desc'].format(**inputs)
    return call_asi_one(prompt)

def email_drafting_agent(inputs):
    prompt = PROMPTS['email_drafting_desc'].format(**inputs)
    return call_asi_one(prompt)

@app.route("/generate_email", methods=["POST"])
def generate_email():
    data = request.json
    required_fields = [
        "company_name", "company_description", "campaign_description",
        "company_rep_name", "company_rep_role", "company_rep_email",
        "prospect_company_name", "prospect_rep_name", "prospect_rep_role",
        "prospect_rep_email"
    ]
    missing = [field for field in required_fields if not data.get(field)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    inputs = {key: data[key] for key in required_fields}
    inputs["today_date"] = date.today().strftime("%Y-%m-%d")

    try:
        # Run agents sequentially
        subject = data_enrichment_agent(inputs)
        insight = needs_analysis_agent(inputs)
        body = email_drafting_agent(inputs)

        full_email = f"""Subject: {subject}\n\n{insight}\n\n{body}"""

        return jsonify({
            "sender_email": data["company_rep_email"],
            "sender_name": data["company_rep_name"],
            "prospect_name": data["prospect_rep_name"],
            "prospect_email": data["prospect_rep_email"],
            "prospect_company_name": data["prospect_company_name"],
            "subject": subject,
            "body": full_email.strip()
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"ASI:One API call failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
