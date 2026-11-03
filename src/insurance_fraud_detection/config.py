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
BIN_ORDERS = {
    "vehicle_price": ["less than 20000", "20000 to 29000", "30000 to 39000",
                      "40000 to 59000", "60000 to 69000", "more than 69000"],
    "age_of_vehicle": ["new", "2 years", "3 years", "4 years",
                       "5 years", "6 years", "7 years", "more than 7"],
    "age_of_policy_holder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30",
                             "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "past_number_of_claims": ["none", "1", "2 to 4", "more than 4"],
    "number_of_suppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "address_change_claim": ["under 6 months", "1 year", "2 to 3 years",
                             "4 to 8 years", "no change"],
    "days_policy_accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "days_policy_claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "number_of_cars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}