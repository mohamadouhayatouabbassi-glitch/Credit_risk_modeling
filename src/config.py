from pathlib import Path

RANDOM_STATE = 42

DATA_PATH = Path("data") / "credit_risk_dataset.csv"

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

FEATURES = ["loan_int_rate", "person_emp_length", "person_income", "loan_amnt"]
TARGET = "loan_status"

DEFAULT_THRESHOLD = 0.35

LGD = 0.6
MARGIN_RATE = 0.08