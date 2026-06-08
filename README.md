# FinSentinel

Stock anomaly detection that combines three different approaches — because no single model catches everything.

The core idea: Isolation Forest handles statistical outliers well but ignores time. LSTM Autoencoder captures temporal patterns but needs a fixed threshold. A DQN agent sits on top and learns to adapt its alerting policy based on what it's seen — instead of a hardcoded cutoff. An agentic layer then translates raw scores into something readable.

Built as an end-to-end pipeline on public market data via `yfinance`, surfaced through a Streamlit dashboard.

---

## How it works

**Isolation Forest** — flags points that are statistically isolated from the rest of the distribution. Fast, no labels needed.

**LSTM Autoencoder** — trained to reconstruct normal price/volume sequences. High reconstruction error = anomaly. Catches regime shifts and unusual patterns over time.

**DQN Agent** — learns an alert policy (flag / hold / clear) from the ensemble's outputs. Adapts to market conditions rather than using static thresholds.

**Agentic layer** — uses the Groq API to generate analyst-style summaries from the pipeline output.

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

Or open `FinSentinel_Colab.ipynb` directly in Google Colab.

---

## Files

```
FinSentinel/
├── FinSentinel_Colab.ipynb   # Full pipeline notebook
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```
