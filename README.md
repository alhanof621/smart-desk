# Smart Desk: IT Service Desk Intelligence

An end-to-end system for automated IT support ticket analysis: NLP issue classification, team routing, resolution time prediction, and anomaly detection, presented in an interactive Streamlit dashboard.

**Team:** Alhanouf Abdullah Alobaid · Sara Mohammed Alshahrani · Raghad Suliman Albalawi · Wojood Turki Almalki
**Program:** Saudi Digital Academy · WeCloudData Data Science Bootcamp · 2026

---

## Business Question

How can organizations use NLP to classify issues, route tickets to the right team, predict resolution times, and detect anomalous ticket surges before they disrupt business?

---

## Dataset

- **Source:** `synthetic_it_support_tickets.csv` (uploaded as `synthetic_it_support_tickets_.csv.zip`; unzip it before running the notebook)
- **Size:** 100,000 tickets × 19 columns, from January 2022 to December 2025 (1,460 days)
- **Text:** only 96 unique message templates
- **Targets:**
  - `issue_type`: 8 classes, for issue classification
  - `product_area`: 7 classes, for team routing
  - `resolution_time_hours`: regression

### Data Cleaning

| Check | Result / Action |
|---|---|
| Duplicate IDs, negative resolution times, invalid status | 0 found |
| CSAT = 0 (29.9%) | Means "not rated"; set to missing (NaN) |
| Missing region (19,997) | Filled with "Unknown" |
| Missing resolution summary (39,887) | Filled with "Pending" |
| Missing resolution time (39,887) | Kept: open, in-progress or on-hold tickets |

---

## Approach

1. **Clean and explore:** quality checks, missing values, outliers, EDA
2. **Engineer features:** time, message, and customer features; one-hot encoding
3. **Process text:** NLTK cleaning, then TF-IDF (unigrams and bigrams)
4. **Model and validate:** compare models per task, using grouped splits to test on unseen messages
5. **Deliver:** Streamlit dashboard with live classification and routing

**Leakage prevention:** routing uses only fields known when a ticket opens; status, CSAT, and reopen data were removed from the resolution-time model; agent replies were excluded from classification.

---

## Results

| Task | Selected Model | Key Result |
|---|---|---|
| Issue type | TF-IDF + Random Forest | 100% on random split; **Macro F1 0.605 ± 0.107** on unseen messages (5-fold GroupKFold) |
| Team routing | TF-IDF + Logistic Regression (text) | 57.4% accuracy (baseline 14.4%); **71.6% top-3 accuracy**; Macro F1 0.516 ± 0.112 on unseen messages |
| Text representation | TF-IDF | Matched Sentence-BERT (all-MiniLM-L6-v2) at lower cost |
| Resolution time | Gradient Boosting (MAE loss) | MAE ≈ 15.2 h, equal to the median baseline; no model beat it |
| Anomaly detection | Isolation Forest + LOF agreement | 45 high-confidence days; 10/10 injected surges detected; 12 strong spikes (z ≥ 3) |

### Key Findings

- **100% accuracy was misleading:** it came from memorizing the 96 templates. Grouped splits give the realistic estimate.
- **Routing is limited by the data:** generic messages can belong to several teams, so top-3 suggestions are more useful than a single prediction.
- **Resolution time is not predictable** with the available features: mutual information of every feature is below 0.008.
- **BERT did not outperform TF-IDF**, because messages are short and keyword-driven.

---

## Dashboard

The Streamlit dashboard has four pages:

- **Overview:** KPIs (total and open tickets, median resolution time, average CSAT, reopen rate) and ticket distributions by team, issue type, hour, and status
- **Ticket Classifier:** type a message to get the predicted issue type and the top-3 suggested teams
- **Anomaly Monitor:** choose a detection method (Isolation Forest, LOF, or both), view flagged days, and see which issue types spiked
- **Model Performance:** model comparison, key findings, and evaluation charts

---

## How to Run the Dashboard

**First time only:**

```bash
cd dashboard_files1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Every time after that:**

```bash
cd dashboard_files1
source venv/bin/activate
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Press `Ctrl + C` in the terminal to stop it.

> **Note:** scikit-learn is pinned to **1.6.1** in `requirements.txt` to match the version used to save the models. A different version can cause errors when loading the `.pkl` files.

---

## Project Structure

```
├── ITservicefinal.ipynb            # Full pipeline: cleaning, EDA, modeling, evaluation
├── synthetic_it_support_tickets_.csv.zip  # Raw dataset (compressed)
├── Smart Desk Report.pdf           # Project report
├── Smart Desk_Presentation.pptx    # Final presentation
└── dashboard_files1/
    ├── app.py                      # Streamlit dashboard
    ├── requirements.txt            # Python dependencies
    ├── final_data.csv              # Cleaned dataset
    ├── issue_classifier.pkl        # Issue type model
    ├── team_routing_model.pkl      # Team routing model
    ├── tfidf.pkl                   # TF-IDF vectorizer
    ├── label_encoder.pkl           # Label encoder
    ├── anomaly_daily_results.csv   # Daily anomaly results
    ├── anomaly_by_issue_type_results.csv
    ├── team_routing_results.csv
    └── *.png                       # Evaluation charts
```

---

## Limitations

- Synthetic data with mostly uniform distributions
- Only 96 unique messages, so results on unseen text vary by about ±0.11
- No priority labels, so priority assignment was dropped from the original proposal
- Generic messages limit routing accuracy
- Anomaly counts depend on the fixed 5% contamination threshold

## Next Steps

- Test the pipeline on real, varied ticket data
- Fine-tune BERT once enough diverse text exists
- Add workload and agent features for resolution time
- Deploy the dashboard and connect it to live ticket data
