# FinSentinel

Stock market anomaly detection pipeline combining Isolation Forest, LSTM Autoencoder, and a DQN alert policy across 15 US equities spanning six market sectors. Trained on 2020–2023, evaluated on held-out 2024 data. Includes an LLM-based agentic layer for automated analyst-style report generation.

---

## Architecture

| Layer | Method | Role |
|-------|--------|------|
| ML | Isolation Forest (contamination=0.05) | Unsupervised anomaly scoring |
| DL | LSTM Autoencoder (seq_len=20, latent_dim=32) | Temporal reconstruction-error scoring |
| Ensemble | 0.40 × IF + 0.60 × LSTM | Fused anomaly score |
| RL | DQN (net_arch=[256,256,128], 150k timesteps) | Learned selective alert policy |
| Agentic | Groq + Llama 3.3 70B | Analyst-style report generation |

---

## Key Design Decisions

**Volatility-adjusted ground truth.** A timestep is labelled anomalous if the 5-day forward return exceeds `max(0.02, volatility_20d × 1.5)`. This adapts the anomaly definition to each ticker's own volatility regime rather than applying a uniform threshold across all stocks.

**RL alert policy.** The DQN learns when to issue an alert given a 6-dimensional state: ensemble score, IF score, LSTM score, log return, volume ratio, normalised RSI. The reward function penalises false alarms proportionally to the agent's running alert rate, which directly discourages the degenerate all-alert policy.

**Chronological split.** Hard boundary at 2023-12-31. No data from the test period (2024) touches training in any component.

---

## Results

Evaluated on 15 tickers across six sectors: Technology (AAPL, MSFT, GOOGL, AMZN), Semiconductors (NVDA, AMD), Financials (JPM, GS), Healthcare (JNJ, PFE), Energy (XOM, CVX), Consumer/Media (NFLX, DIS).

**Baseline comparison — test split 2024:**

| Method | Avg Precision | Avg Recall | Avg F1 |
|--------|--------------|------------|--------|
| Z-Score Baseline | 0.494 | 0.026 | 0.049 |
| IF Only | 0.381 | 0.040 | 0.072 |
| FinSentinel (Ensemble+RL) | 0.482 | 0.380 | 0.417 |

5.8× F1 improvement over the strongest baseline.

**Ablation study — mean F1 by pipeline depth:**

| Stage | Mean F1 |
|-------|---------|
| IF Only | 0.072 |
| Ensemble without RL | 0.061 |
| Full Pipeline | 0.417 |

**Event-anchored validation:** 70.7% mean hit rate across 15 tickers on 10 known 2024 market events (±3 trading day window, no event supervision). XOM 10/10, AAPL 9/10, PFE 9/10.

**Bootstrap 95% CIs** computed over 1,000 resamples. All intervals strictly above zero. Mean F1: 0.416.

---

## Notebooks

| File | Purpose |
|------|---------|
| `FinSentinel_1_Training.ipynb` | Full pipeline training — data, features, IF, LSTM, DQN. Saves models and artifacts to Google Drive. Run once. |
| `FinSentinel_2_Evaluation.ipynb` | Loads from Drive, runs evaluation, ablation, statistical significance, event validation, agentic layer. No retraining required. |

---

## Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `stable-baselines3` · `yfinance` · `Groq API` · `Plotly`

---

## Paper

Submitted to Expert Systems with Applications (Elsevier). Preprint available via SSRN on submission.

---

## Setup

```bash
git clone https://github.com/kapurV06/FinSentinel
cd FinSentinel
pip install -r requirements.txt
```

Open `FinSentinel_1_Training.ipynb` in Google Colab with GPU runtime. Add your `GROQ_API_KEY` to Colab secrets. Run Notebook 1 once, then use Notebook 2 for all evaluation.
