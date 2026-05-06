import numpy as np


def profit_per_loan(
    y_true,
    accepted,
    loan_amnt,
    lgd: float,
    margin_rate: float,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    accepted = np.asarray(accepted).astype(bool)
    loan_amnt = np.asarray(loan_amnt)

    profit = np.zeros_like(loan_amnt, dtype=float)

    profit[accepted & (y_true == 0)] = margin_rate * loan_amnt[accepted & (y_true == 0)]
    profit[accepted & (y_true == 1)] = -lgd * loan_amnt[accepted & (y_true == 1)]

    return profit