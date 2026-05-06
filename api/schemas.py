from pydantic import BaseModel
from typing import Optional


class CreditApplication(BaseModel):
    loan_int_rate: Optional[float] = None
    person_emp_length: Optional[float] = None
    person_income: Optional[float] = None
    loan_amnt: Optional[float] = None


class PredictionOut(BaseModel):
    prob_default: float
    decision: str
    threshold: float