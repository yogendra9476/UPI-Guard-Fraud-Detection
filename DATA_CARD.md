# Data Card: `upi_transactions_2024.csv`

| Item | Value |
|---|---|
| File | `data/upi_transactions_2024.csv` (28.4 MB) |
| SHA-256 | see `artifacts/manifest.json` → `dataset_sha256` |
| Rows × columns | 250,000 × 17 |
| Period | 2024-01-01 00:05 to 2024-12-30 23:55 |
| Fraud label | `fraud_flag`; 480 positives (0.192 %) |
| Missing values / duplicate rows / duplicate IDs | 0 / 0 / 0 |
| Customer / account / device identifiers | **None** |

## Provenance
The file is a publicly redistributed transaction log. It was **not** issued by NPCI, the RBI or
any bank or PSP, and no generation or collection procedure is documented with it. Our audit
(`artifacts/audit_univariate.csv`, `artifacts/audit_permutation.json`) finds no statistically
detectable association between `fraud_flag` and any recorded attribute. That is what one would
expect if the label had been assigned independently of the features, as happens in a simulated
dataset. **We therefore treat the file as a public, probably synthetic benchmark and make no
claims about real UPI fraud from it.**

## Columns and pre-debit availability
| Column | Type | Known before debit? | Used as model input |
|---|---|---|---|
| transaction id | ID | yes | no (identifier) |
| timestamp | datetime | yes | ordering, split and velocity only |
| transaction type | cat (4) | yes | yes |
| merchant_category | cat (10) | yes | yes |
| amount (INR) | int, 10 to 42,099 | yes | yes (log and train-fitted ratios) |
| transaction_status | SUCCESS/FAILED | **no**, an authorisation outcome | **no (leakage)** |
| sender_age_group, receiver_age_group | cat (5) | yes (KYC) | yes |
| sender_state | cat (10) | yes | yes |
| sender_bank, receiver_bank | cat (8) | yes (VPA resolution) | yes |
| device_type | cat (3) | yes | yes |
| network_type | cat (4) | yes | yes |
| hour_of_day, day_of_week, is_weekend | derived from timestamp | yes | yes |
| fraud_flag | target | no (confirmed later) | target |

## Intended use
Teaching and benchmarking of *evaluation methodology*: leakage control, out-of-time testing,
calibration, cost-sensitive thresholds and learnability auditing.

## Not suitable for
Estimating real UPI fraud rates or losses, or training a model for production authorisation.

## Known limitations
No payer or payee identity, so true behavioural velocity, payee novelty and device history
cannot be computed. The segment-level velocity proxies are coarse. The label's confirmation
delay is unknown.
