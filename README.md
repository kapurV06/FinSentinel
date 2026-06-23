# FinSentinel

Stock anomaly detection pipeline for equities. Combines Isolation Forest, LSTM Autoencoder, and a DQN alert agent — trained on 2020–2023, evaluated on held-out 2024 data.

---

## Architecture

| Layer | Method | Role |
|-------|--------|------|
| ML | Isolation Forest | Unsupervised anomaly scoring |
| DL | LSTM Autoencoder | Temporal reconstruction-error scoring |
| Ensemble | 0.4 · IF + 0.6 · LSTM | Fused anomaly score |
| RL | DQN (Stable-Baselines3) | Learned alert policy |
| Agentic | Groq + Llama 3.3 70B | Analyst-style report generation |

---

## Results

Evaluated on 5 tickers (AAPL, TSLA, AMZN, MSFT, GOOGL) — test split 2024-01-01 to 2024-12-31.

**DQN agent — out-of-sample performance:**

| Ticker | Precision | Recall | F1 | Alert Rate |
|--------|-----------|--------|----|------------|
| AAPL | 0.551 | 0.970 | 0.703 | 95.5% |
| TSLA | 0.796 | 1.000 | 0.886 | 100.0% |
| AMZN | 0.580 | 1.000 | 0.734 | 100.0% |
| MSFT | 0.550 | 0.858 | 0.671 | 85.3% |
| GOOGL | 0.659 | 0.747 | 0.700 | 69.4% |

LSTM Autoencoder val_loss: 0.425 (AAPL) · 0.399 (TSLA) · 0.411 (AMZN) · 0.448 (MSFT) · 0.508 (GOOGL)

**Known limitation:** alert rate on TSLA and AMZN is 100% — the DQN hasn't learned selectiveness on those tickers. FP penalty is -2.5; may need further tuning.

---

## Evaluation protocol

80/20 chronological split. IF and LSTM trained on full 2020–2023 window. DQN trained on 2020–2023 environment, weights frozen for 2024 evaluation — no leakage.

---

## Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `stable-baselines3` · `yfinance` · `Plotly` · `Streamlit` · `Groq API`

---

## Run it

```bash
git clone https://github.com/kapurV06/FinSentinel
cd FinSentinel
pip install -r requirements.txt
streamlit run app.py
```

Or open `FinSentinel_Colab.ipynb` in Google Colab.

---

## Files

```
FinSentinel/
├── FinSentinel_Colab.ipynb   # Full pipeline notebook
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```
