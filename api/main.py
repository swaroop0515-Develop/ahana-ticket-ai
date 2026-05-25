import re
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


# -----------------------------
# LOAD VECTORIZER
# -----------------------------

vectorizer = joblib.load(
    "../models/vectorizer.pkl"
)


# -----------------------------
# LOAD MODELS
# -----------------------------

models = {}

model_files = {
    "Impact": "../models/impact_model.pkl",
    "Urgency": "../models/urgency_model.pkl",
    "Priority": "../models/priority_model.pkl",
    "Category": "../models/category_model.pkl",
    "Subcategory": "../models/subcategory_model.pkl",
    "Group": "../models/group_model.pkl",
    "Request Type": "../models/request_type_model.pkl",
    "Environment Type": "../models/environment_type_model.pkl"
}

for field, path in model_files.items():

    try:

        models[field] = joblib.load(path)

        print(f"{field} model loaded")

    except:

        print(f"{field} model not found")


# -----------------------------
# FASTAPI APP
# -----------------------------

app = FastAPI()


# -----------------------------
# INPUT MODEL
# -----------------------------

class TicketRequest(BaseModel):

    subject: str
    description: str
    account: str


# -----------------------------
# CLEAN TEXT
# -----------------------------

def clean_text(text):

    text = str(text)

    text = re.sub(r'[\n\r\t]+', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------

def predict_ticket(subject, description, account):

    subject = clean_text(subject)

    description = clean_text(description)

    account = clean_text(account)

    text = f"{subject} {description} {account}"

    vector = vectorizer.transform([text])

    predictions = {}

    for field, model in models.items():

        prediction = model.predict(vector)[0]

        predictions[field] = prediction

    return predictions


# -----------------------------
# API ENDPOINT
# -----------------------------

@app.post("/predict")

def predict(request: TicketRequest):

    result = predict_ticket(
        request.subject,
        request.description,
        request.account
    )

    return result