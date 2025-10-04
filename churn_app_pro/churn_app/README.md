# Churn Retention Cockpit (Professional Streamlit App)

A complete, production-style demo for customer churn prediction on the public **BankChurners** dataset.

## Structure
```
churn_app/
├─ train_model.py                 # train & calibrate; saves model + plots to app/
├─ app/
│  ├─ app.py                      # Streamlit UI (Home, EDA, Predictor, Insights, Model Card)
│  ├─ utils.py                    # helpers: schema, derived fields, reasons, ROI thresholding
│  ├─ styles.css                  # UI polish
│  ├─ model_card.md               # documentation
│  └─ assets/                     # roc.png, pr.png, calibration.png, feature_importance.png (after training)
└─ sample_batch.csv               # header example for batch scoring
```

## Quick start
1. Place `BankChurners.csv` in `churn_app/` (root) or in `churn_app/app/`.
2. Install deps (in a venv is recommended):
   ```bash
   pip install streamlit scikit-learn xgboost pandas numpy matplotlib joblib
   ```
3. Train and generate plots:
   ```bash
   python train_model.py
   ```
4. Run the UI:
   ```bash
   streamlit run app/app.py
   ```

- Use **Predictor → Batch Scoring** to upload a CSV and download a ranked retention queue.
- Use **Single Customer** to tweak 6–10 key drivers (derived fields auto-computed).
- See **Insights** for ROC/PR/Calibration and top feature importances.
- See **Model Card** for documentation and governance notes.
