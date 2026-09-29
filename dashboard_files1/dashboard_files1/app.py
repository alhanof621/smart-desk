# ============================================================
# IT Service Desk Intelligence - Streamlit Dashboard
# التشغيل:  streamlit run app.py
# ============================================================
import os
import re
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="IT Service Desk Intelligence", page_icon="🎫", layout="wide")

# خلوها True إذا شغلتوا خلية Text Preprocessing (✨) قبل تدريب موديل issue_type
# وخلوها False إذا الموديل تدرب على النص الخام
USE_CLEAN_TEXT = True


# ------------------------------------------------------------
# تحميل الداتا والموديلات (مرة وحدة فقط بفضل الـ cache)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("final_data.csv", parse_dates=["created_at"])
    daily = pd.read_csv("anomaly_daily_results.csv", parse_dates=["created_at"])
    by_type = pd.read_csv("anomaly_by_issue_type_results.csv", parse_dates=["created_at"])
    routing = pd.read_csv("team_routing_results.csv")
    return df, daily, by_type, routing


@st.cache_resource
def load_models():
    return {
        "tfidf": joblib.load("tfidf.pkl"),
        "issue_clf": joblib.load("issue_classifier.pkl"),
        "le_issue": joblib.load("label_encoder.pkl"),
        "router": joblib.load("team_routing_model.pkl"),
    }


@st.cache_resource
def get_text_cleaner():
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english")) - {"not", "no", "how"}
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        tokens = re.findall(r"[a-z]+", str(text).lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
        return " ".join(tokens)

    return clean_text


df, daily, by_type, routing = load_data()
models = load_models()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("🎫 Service Desk Intelligence")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "🤖 Ticket Classifier", "🚨 Anomaly Monitor", "📈 Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption("WeCloudData Data Science Bootcamp Project")


# ============================================================
# Page 1: Overview
# ============================================================
if page == "📊 Overview":
    st.title("📊 Service Desk Overview")

    # فلتر التاريخ
    min_d, max_d = df["created_at"].min().date(), df["created_at"].max().date()
    date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        dff = df[(df["created_at"] >= start) & (df["created_at"] < end)]
    else:
        dff = df

    # KPIs
    open_statuses = ["open", "in_progress", "on_hold"]
    rated = dff[dff["csat_score"] > 0]          # CSAT = 0 يعني العميل ما قيّم
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Tickets", f"{len(dff):,}")
    c2.metric("Open Tickets", f"{dff['status'].isin(open_statuses).sum():,}")
    c3.metric("Median Resolution", f"{dff['resolution_time_hours'].median():.1f} h")
    c4.metric("Avg CSAT (rated)", f"{rated['csat_score'].mean():.2f} / 5" if len(rated) else "-")
    c5.metric("Reopen Rate", f"{dff['reopened'].mean():.1%}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        area_counts = dff["product_area"].value_counts().reset_index()
        area_counts.columns = ["product_area", "count"]
        fig = px.bar(area_counts, x="product_area", y="count", title="Tickets by Product Area (Team)",
                     color="product_area")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        issue_counts = dff["issue_type"].value_counts().reset_index()
        issue_counts.columns = ["issue_type", "count"]
        fig = px.bar(issue_counts, x="count", y="issue_type", orientation="h",
                     title="Tickets by Issue Type", color="issue_type")
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        hourly = dff.groupby(dff["created_at"].dt.hour).size().reset_index(name="count")
        hourly.columns = ["hour", "count"]
        fig = px.bar(hourly, x="hour", y="count", title="Ticket Volume by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        status_counts = dff["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(status_counts, names="status", values="count", title="Ticket Status", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    weekly = dff.set_index("created_at").resample("W").size().reset_index(name="count")
    weekly = weekly.iloc[1:-1]   # نشيل أول وآخر أسبوع لأنهم ناقصين (أسابيع جزئية)
    fig = px.line(weekly, x="created_at", y="count", title="Weekly Ticket Volume")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 2: Ticket Classifier
# ============================================================
elif page == "🤖 Ticket Classifier":
    st.title("🤖 Classify & Route a New Ticket")
    st.write("Type a ticket message and the system predicts the **issue type** and the **team** it should go to.")

    examples = ["(write your own)"] + sorted(df["initial_message"].dropna().unique().tolist())
    example = st.selectbox("Pick an example message (optional)", examples)
    default_msg = "" if example == "(write your own)" else example
    message = st.text_area("Ticket message", value=default_msg, height=120)

    c1, c2, c3 = st.columns(3)
    channel = c1.selectbox("Channel", sorted(df["channel"].unique()))
    platform = c2.selectbox("Platform", sorted(df["platform"].unique()))
    segment = c3.selectbox("Customer segment", sorted(df["customer_segment"].unique()))

    if st.button("Classify ticket", type="primary"):
        if not message.strip():
            st.warning("Please enter a ticket message.")
        else:
            # 1) Issue type
            text_for_clf = get_text_cleaner()(message) if USE_CLEAN_TEXT else message
            X_txt = models["tfidf"].transform([text_for_clf])
            issue_proba = models["issue_clf"].predict_proba(X_txt)[0]
            issue_idx = int(np.argmax(issue_proba))
            issue_label = models["le_issue"].inverse_transform([models["issue_clf"].classes_[issue_idx]])[0]

            # 2) Team routing (top 3)
            row = pd.DataFrame([{"initial_message": message, "channel": channel,
                                 "platform": platform, "customer_segment": segment}])
            router = models["router"]
            team_proba = router.predict_proba(row)[0]
            top3 = np.argsort(team_proba)[::-1][:3]

            left, right = st.columns(2)
            with left:
                st.subheader("Issue Type")
                st.metric("Predicted", issue_label, f"{issue_proba[issue_idx]:.0%} confidence")
            with right:
                st.subheader("Suggested Teams")
                for rank, i in enumerate(top3, start=1):
                    st.write(f"**{rank}. {router.classes_[i]}** — {team_proba[i]:.0%}")
                    st.progress(float(team_proba[i]))

            if team_proba[top3[0]] < 0.5:
                st.info("Low routing confidence: a human agent should confirm the team from the top-3 suggestions.")


# ============================================================
# Page 3: Anomaly Monitor
# ============================================================
elif page == "🚨 Anomaly Monitor":
    st.title("🚨 Anomaly Monitor")

    method = st.sidebar.radio("Detection method",
                              ["Isolation Forest", "LOF", "Both agree (high confidence)"])
    if method == "Isolation Forest":
        mask = daily["anomaly_score"] == -1
    elif method == "LOF":
        mask = daily["lof_anomaly"] == -1
    else:
        mask = (daily["anomaly_score"] == -1) & (daily["lof_anomaly"] == -1)
    anomalies = daily[mask]

    c1, c2, c3 = st.columns(3)
    c1.metric("Days monitored", f"{len(daily):,}")
    c2.metric("Anomalous days", f"{len(anomalies):,}")
    c3.metric("Share of days", f"{len(anomalies) / len(daily):.1%}")

    metric_labels = {"ticket_count": "Ticket Count",
                     "avg_resolution_time": "Avg Resolution Time (h)",
                     "reopened_rate": "Reopen Rate"}
    metric = st.selectbox("Metric", list(metric_labels), format_func=metric_labels.get)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["created_at"], y=daily[metric], mode="lines",
                             name=metric_labels[metric], line=dict(width=1)))
    fig.add_trace(go.Scatter(x=anomalies["created_at"], y=anomalies[metric], mode="markers",
                             name="Anomaly", marker=dict(color="red", size=8)))
    fig.update_layout(title=f"Daily {metric_labels[metric]} with Detected Anomalies",
                      xaxis_title="Date", yaxis_title=metric_labels[metric])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: a day is flagged using all three metrics together, so a red point can look normal on one metric.")

    st.subheader("Anomalous Days")
    st.dataframe(anomalies.sort_values("ticket_count", ascending=False)
                 [["created_at", "ticket_count", "avg_resolution_time", "reopened_rate"]]
                 .round(2), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Which issue type is spiking?")
    z_min = st.slider("Minimum spike strength (z-score)", 0.0, 6.0, 3.0, 0.5)
    strong = by_type[by_type["spike_z"] >= z_min]
    st.write(f"**{len(strong)}** anomalous days with a spike of z ≥ {z_min}")

    if len(strong):
        counts = strong["spiking_issue_type"].value_counts().reset_index()
        counts.columns = ["issue_type", "days"]
        fig = px.bar(counts, x="issue_type", y="days", title="Spiking Issue Type in Anomalous Days",
                     color_discrete_sequence=["indianred"])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(strong.sort_values("spike_z", ascending=False)
                     [["created_at", "spiking_issue_type", "spike_z"]],
                     use_container_width=True, hide_index=True)


# ============================================================
# Page 4: Model Performance
# ============================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Team Routing — Model Comparison")
    fig = px.bar(routing.sort_values("Macro F1"), x="Macro F1", y="Model", orientation="h",
                 text="Macro F1", title="Macro F1 by Model")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(routing, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Key Findings")
    st.markdown("""
- **Issue type classification** reaches 100% on known message templates.
 The dataset has only 96 unique messages, so we also tested on *unseen* messages (5-fold GroupKFold):
 **Macro F1 = 0.605 ± 0.107**, which is the realistic estimate for new wording.
- **Team routing** is limited by the data: the same message can belong to different teams.
 On unseen messages, **Macro F1 = 0.516 ± 0.112**, so the top-3 suggestions are more useful than a single prediction.
- **Resolution time** could not be predicted better than the median baseline:
  none of the available features carries information about it (confirmed with mutual information).
- **BERT vs TF-IDF**: pre-trained BERT embeddings did not outperform TF-IDF on this dataset,
  because the messages are short templates where keywords already carry the meaning.
- **Anomaly detection** flags unusual days; only strong spikes (z ≥ 3) should trigger alerts.
""")

    # الصور المحفوظة من النوتبوك: كل صورة في تبويب، وبحجم مناسب
    st.markdown("---")
    st.subheader("Evaluation Charts")
    images = {"bert_vs_tfidf.png": "BERT vs TF-IDF",
              "routing_confusion_matrix.png": "Routing Confusion Matrix",
              "issue_type_confusion_matrix.png": "Issue Type Confusion Matrix",
              "resolution_time_mutual_info.png": "Resolution Time (MI)",
              "anomaly_Spiking_issue_type.png": "Anomaly Spikes"}
    available = {path: label for path, label in images.items() if os.path.exists(path)}
    IMG_WIDTH = 750

    if available:
        tabs = st.tabs(list(available.values()))
        for tab, path in zip(tabs, available):
            with tab:
                st.image(path, width=IMG_WIDTH)
                if path == "bert_vs_tfidf.png":
                    st.caption("Note: grouped-split results here come from a single split and may vary by several points. "
                               "The 5-fold GroupKFold results in Key Findings are more reliable.")