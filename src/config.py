"""Central configuration for the UPI Guard pipeline.

Every number that affects a reported result is defined here, so that the
report, the Streamlit app and the training script share one source of truth.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "upi_transactions_2024.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "report" / "figures"

SEED = 42

# ----------------------------------------------------------------------------
# Out-of-time split (by transaction timestamp). The test window is touched once.
# ----------------------------------------------------------------------------
TRAIN_END = "2024-09-01"   # train:      2024-01-01 .. 2024-08-31
VALID_END = "2024-11-01"   # validation: 2024-09-01 .. 2024-10-31
#                            test:       2024-11-01 .. 2024-12-30

# ----------------------------------------------------------------------------
# Raw columns and their availability at the pre-debit decision point.
# ----------------------------------------------------------------------------
TARGET = "fraud_flag"
RAW_RENAME = {
    "transaction id": "transaction_id",
    "transaction type": "transaction_type",
    "amount (INR)": "amount_inr",
}
# Columns that exist before the debit is authorised (used as model inputs)
PRE_DEBIT_CATEGORICAL = [
    "transaction_type", "merchant_category", "sender_age_group",
    "receiver_age_group", "sender_state", "sender_bank", "receiver_bank",
    "device_type", "network_type", "day_of_week",
]
# Columns excluded from modelling, with the reason
EXCLUDED = {
    "transaction_id": "Row identifier; carries no behavioural meaning.",
    "timestamp": "Used only for ordering, splitting and velocity features.",
    "transaction_status": "Outcome of authorisation (SUCCESS/FAILED); not "
                          "known before the debit decision -> temporal leakage.",
}

# ----------------------------------------------------------------------------
# Business cost assumptions (INR). These are explicit assumptions, varied in
# a sensitivity analysis; they are not measured bank costs.
# ----------------------------------------------------------------------------
COST_REVIEW_INR = 25.0        # cost of one step-up / manual review (OTP, ops time, friction)
REVIEW_GRID_INR = [5, 10, 25, 50, 100]
STEP_UP_EFFECTIVENESS = 1.0   # share of a flagged fraud that is prevented (optimistic)

# ----------------------------------------------------------------------------
# Positive-control (semi-synthetic) experiment
# ----------------------------------------------------------------------------
SIGNAL_STRENGTHS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
PC_REPLICATES = 3
PC_CNN_STRENGTH = 1.5
CNN_ORDERINGS = 5

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 200
