# Early-Warnings-in-Higher-Education

Predicts STEM student outcomes (Dropout / Enrolled / Graduate) to flag at-risk students early. Trains Logistic Regression and Decision Tree models, evaluates them with confusion matrices and SHAP feature importance, and serves an interactive risk predictor via Streamlit.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

1. **Train the models** (creates `logistic_model.pkl`, `decision_tree_model.pkl`, `scaler.pkl`):

   ```bash
   python model.py
   ```

2. **Evaluate** — prints classification reports and writes confusion matrix + SHAP plots to `plots/`:

   ```bash
   python evaluate.py
   ```

3. **Launch the app** — EDA dashboard and interactive risk predictor:

   ```bash
   streamlit run app.py
   ```

> Run `python model.py` first: `evaluate.py` and `app.py` both load the trained `.pkl` artifacts, which are not checked into git.
