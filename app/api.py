from fastapi import FastAPI
from pydantic import BaseModel

from insurance_fraud_detection.predict import predict_one

SAMPLE = {
    "base_policy": "Liability",
    "fault": "Policy Holder",
    "make": "Chevrolet",
    "past_number_of_claims": "none",
    "address_change_claim": "no change",
    "deductible": 400,
    "vehicle_price": "20000 to 29000",
    "month": "Jul",
    "month_claimed": "Jul",
    "age_of_policy_holder": "36 to 40",
    "number_of_suppliments": "none",
    "agent_type": "External",
    "age_of_vehicle": "7 years",
}


class Claim(BaseModel):
    base_policy: str
    fault: str
    make: str
    past_number_of_claims: str
    address_change_claim: str
    deductible: int
    vehicle_price: str
    month: str
    month_claimed: str
    age_of_policy_holder: str
    number_of_suppliments: str
    agent_type: str
    age_of_vehicle: str

    model_config = {"json_schema_extra": {"examples": [SAMPLE]}}


app = FastAPI(title="Insurance Fraud Detection API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(claim: Claim) -> dict:
    return predict_one(claim.model_dump())