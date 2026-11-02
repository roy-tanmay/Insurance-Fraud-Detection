from pathlib import Path

# Path
ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "fraud_oracle.csv"

# Data
TARGET = "fraud_found_p"
SEED = 42
TEST_SIZE = 0.20

print(RAW_CSV)