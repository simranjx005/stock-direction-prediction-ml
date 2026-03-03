import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# ---------------- Page Config ----------------
st.set_page_config(page_title="Stock Direction Predictor", layout="wide")

st.title("📈 Stock Direction Prediction - Algorithm Comparison")
st.markdown("Compare **Naive Bayes, Logistic Regression, and Random Forest**")
st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

run_button = st.sidebar.button("🚀 Run Model")

# ---------------- Main Logic ----------------
if run_button:

    # Download data
    data = yf.download(
        symbol.strip().upper(),
        start=str(start_date),
        end=str(end_date)
    )

    if len(data) < 60:
        st.error("Please select a longer date range (at least 3 months).")
        st.stop()

    # -------- Feature Engineering --------
    data['Return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()

    data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)
    data = data.dropna()

    X = data[['Return', 'MA10', 'MA50']]
    y = data['Target']

    # -------- Train/Test Split --------
    split_index = int(len(data) * 0.8)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # -------- Scaling (Important for LR & NB) --------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------- Models --------
    models = {
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42)
    }

    scores = {}
    predictions = {}

    with st.spinner("Training All Models..."):
        for name, model in models.items():
            
            # Use scaled data for LR and NB
            if name in ["Naive Bayes", "Logistic Regression"]:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores[name] = acc
            predictions[name] = y_pred

    # -------- Display Scores --------
    st.markdown("### 📊 Algorithm Accuracy Comparison")

    score_df = pd.DataFrame({
        "Algorithm": scores.keys(),
        "Accuracy (%)": [round(v*100, 2) for v in scores.values()]
    }).sort_values(by="Accuracy (%)", ascending=False)

    # -------- Clean Algorithm Comparison Layout --------
    st.markdown("### 📊 Algorithm Performance Comparison")

    col1, col2, col3 = st.columns(3)

    algorithms = list(scores.keys())
    accuracies = list(scores.values())

    with col1:
        st.metric(
            label="🧠 Naive Bayes",
            value=f"{scores['Naive Bayes']*100:.2f}%"
    )
    with col2:
        st.metric(
            label="📊 Logistic Regression",
            value=f"{scores['Logistic Regression']*100:.2f}%"
    )
    with col3:
        st.metric(
            label="🌳 Random Forest",
            value=f"{scores['Random Forest']*100:.2f}%"
    )

    # -------- Best Model --------
    best_model_name = max(scores, key=scores.get)
    best_accuracy = scores[best_model_name]

    st.success(f"🏆 Best Algorithm: **{best_model_name}** ({best_accuracy*100:.2f}%)")

    # -------- Confusion Matrix of Best Model --------
    st.markdown("### 📊 Confusion Matrix (Best Model)")

    best_predictions = predictions[best_model_name]

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, best_predictions)).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # -------- Actual vs Predicted Graph --------
    st.markdown("### 📈 Actual vs Predicted Market Growth Curve")
    
    actual_binary = y_test.values
    predicted_binary = best_predictions
    
    # Convert direction to +1 / -1
    actual_direction = np.where(actual_binary == 1, 1, -1)
    predicted_direction = np.where(predicted_binary == 1, 1, -1)
    
    # Take last 100 points
    actual_direction = actual_direction[-100:]
    predicted_direction = predicted_direction[-100:]
    
    # Create cumulative growth curve
    actual_growth = np.cumsum(actual_direction)
    predicted_growth = np.cumsum(predicted_direction)
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.plot(actual_growth,
    linewidth=3,
    label="Actual Market Growth")
    
    ax.plot(predicted_growth,
        linewidth=3,
        linestyle="--",
        label="Predicted Strategy Growth")
    
    ax.set_title(f"Market Growth Simulation - {best_model_name}",
             fontsize=16,
             fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Growth")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # -------- Latest Prediction --------
    st.markdown("### 📌 Latest Prediction (Using Best Model)")

    latest_features = X.tail(1)

    if best_model_name in ["Naive Bayes", "Logistic Regression"]:
        latest_features = scaler.transform(latest_features)

    final_model = models[best_model_name]
    latest_prediction = final_model.predict(latest_features)[0]

    if latest_prediction == 1:
        st.success("📈 STOCK WILL GO UP")
    else:
        st.error("📉 STOCK WILL GO DOWN")

    # -------- Raw Data --------
    with st.expander("📂 Show Raw Data"):
        st.dataframe(data)