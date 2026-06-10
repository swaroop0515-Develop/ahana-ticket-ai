import re
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =====================================================
# LOAD VECTORIZER
# =====================================================

vectorizer = joblib.load(
    "models/vectorizer.pkl"
)


# =====================================================
# LOAD MODELS
# =====================================================

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
        print(f"❌ Failed to load {field}: {e}")


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Ticket Classification API",
    version="2.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================================
# REQUEST MODEL
# =====================================================

class TicketRequest(BaseModel):
    subject: str
    description: str
    account: str


# =====================================================
# TEXT CLEANING
# =====================================================

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


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_ticket(
    subject,
    description,
    account
):

    subject = clean_text(subject)
    description = clean_text(description)
    account = clean_text(account)

    combined_text = (
        f"{subject} "
        f"{description} "
        f"{account}"
    )

    vector = vectorizer.transform(
        [combined_text]
    )

    predictions = {}

    confidence_scores = []

    for field, model in models.items():

        try:

            prediction = model.predict(
                vector
            )[0]

            confidence = 0.0

            # Get confidence if supported
            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    vector
                )[0]

                confidence = round(
                    float(max(probabilities)) * 100,
                    2
                )

            predictions[field] = str(
                prediction
            )

            predictions[
                f"{field}Confidence"
            ] = confidence

            confidence_scores.append(
                confidence
            )

        except Exception as e:

            predictions[field] = None

            predictions[
                f"{field}Confidence"
            ] = 0.0

            print(
                f"Prediction failed for {field}: {e}"
            )

    # =================================================
    # OVERALL CONFIDENCE
    # =================================================

    if confidence_scores:

        predictions[
            "OverallConfidence"
        ] = round(
            sum(confidence_scores)
            / len(confidence_scores),
            2
        )

        predictions[
            "MinimumConfidence"
        ] = round(
            min(confidence_scores),
            2
        )

    else:

        predictions[
            "OverallConfidence"
        ] = 0.0

        predictions[
            "MinimumConfidence"
        ] = 0.0

    return predictions


# =====================================================
# ROOT ENDPOINT
# =====================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Ticket Classification API"
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": len(models),
        "model_names": list(
            models.keys()
        )
    }


# =====================================================
# PREDICT ENDPOINT
# =====================================================

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
