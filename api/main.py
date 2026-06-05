import re
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ----------------------------------
# LOAD VECTORIZER
# ----------------------------------

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)


# ----------------------------------
# LOAD MODELS
# ----------------------------------

models = {}

model_files = {
    "Impact": "models/impact_model.pkl",
    "Urgency": "models/urgency_model.pkl",
    "Priority": "models/priority_model.pkl",
    "Category": "models/category_model.pkl",
    "Subcategory": "models/subcategory_model.pkl",
    "Group": "models/group_model.pkl",
    "RequestType": "models/request_type_model.pkl",
    "EnvironmentType": "models/environment_type_model.pkl"
}

for field, path in model_files.items():

    try:

        models[field] = joblib.load(path)

        print(f"✅ {field} model loaded")

    except Exception as e:

        print(
            f"❌ {field} model not found: {e}"
        )


# ----------------------------------
# FASTAPI APP
# ----------------------------------

app = FastAPI(
    title="Ticket Classification API",
    version="1.0"
)


# ----------------------------------
# CORS
# ----------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "*"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ----------------------------------
# INPUT MODEL
# ----------------------------------

class TicketRequest(BaseModel):

    subject: str

    description: str

    account: str


# ----------------------------------
# CLEAN TEXT
# ----------------------------------

def clean_text(text):

    text = str(text)

    text = re.sub(
        r'[\n\r\t]+',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ----------------------------------
# PREDICT TICKET
# ----------------------------------

def predict_ticket(
    subject,
    description,
    account
):

    subject = clean_text(
        subject
    )

    description = clean_text(
        description
    )

    account = clean_text(
        account
    )

    combined_text = (
        f"{subject} "
        f"{description} "
        f"{account}"
    )

    vector =
        vectorizer.transform(
            [combined_text]
        )

    predictions = {}

    for field, model in models.items():

        prediction =
            model.predict(
                vector
            )[0]

        predictions[
            field
        ] = str(
            prediction
        )

    return predictions


# ----------------------------------
# HEALTH CHECK
# ----------------------------------

@app.get("/")

def root():

    return {
        "status": "running",
        "message":
        "Ticket Classification API"
    }


# ----------------------------------
# PREDICTION ENDPOINT
# ----------------------------------

@app.post("/predict")

def predict(
    request: TicketRequest
):

    result = predict_ticket(
        request.subject,
        request.description,
        request.account
    )

    return result
