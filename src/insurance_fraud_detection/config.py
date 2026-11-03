from pathlib import Path

# Path
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "fraud_oracle.csv"

# Data
TARGET = "fraud_found_p"
SEED = 42
TEST_SIZE = 0.20
DROP_COLUMNS = [
    "policy_number",
    "rep_number"
]

# Columns
POOL_MAP = {
    "days_policy_accident": {
        "none": "30 or fewer",
        "1 to 7": "30 or fewer",
        "8 to 15": "30 or fewer",
        "15 to 30": "30 or fewer",
    },
    "days_policy_claim": {
        "8 to 15": "30 or fewer",
        "15 to 30": "30 or fewer",
    },
    "address_change_claim": {
        "under 6 months": "1 year or less",
        "1 year": "1 year or less",
    },
    "number_of_cars": {
        "3 to 4": "3 or more vehicles",
        "5 to 8": "3 or more vehicles",
        "more than 8": "3 or more vehicles",
    },
    "age_of_policy_holder": {
        "16 to 17": "16 to 20",
        "18 to 20": "16 to 20",
    },
    "age_of_vehicle": {
        "2 years": "2 to 3 years",
        "3 years": "2 to 3 years",
    },
    "vehicle_price": {
        "40000 to 59000": "40000 to 69000",
        "60000 to 69000": "40000 to 69000",
    },
}

BIN_ORDERS = {
    "days_policy_accident": ["30 or fewer", "more than 30"],
    "days_policy_claim": ["30 or fewer", "more than 30"],
    "address_change_claim": ["1 year or less", "2 to 3 years", "4 to 8 years", "no change"],
    "number_of_suppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "number_of_cars": ["1 vehicle", "2 vehicles", "3 or more vehicles"],
    "age_of_policy_holder": ["16 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40",
                             "41 to 50", "51 to 65", "over 65"],
    "age_of_vehicle": ["new", "2 to 3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"],
    "vehicle_price": ["less than 20000", "20000 to 29000", "30000 to 39000",
                      "40000 to 69000", "more than 69000"],
    "past_number_of_claims": ["none", "1", "2 to 4", "more than 4"],
}