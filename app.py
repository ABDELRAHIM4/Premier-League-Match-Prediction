import streamlit as st
import joblib
import pandas as pd

# Load model + team encoders
model = joblib.load("model.pkl")
le_home = joblib.load("le_home.pkl")
le_away = joblib.load("le_away.pkl")

st.title("⚽ Premier League Match Predictor")

# Team selection
teams = list(le_home.keys())
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

# Match details
date = st.date_input("Match Date")
wk = st.number_input("Gameweek", min_value=1, step=1)

if st.button("Predict Result"):
    try:
        if home_team not in le_home or away_team not in le_away:
            st.error("❌ Team not found in training data")
        else:
            home_code = le_home[home_team]
            away_code = le_away[away_team]

            X = [[home_code, 0, 0, away_code, wk]]

            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]

            label_map = {
                0: f"{away_team} Win",
                1: "Draw",
                2: f"{home_team} Win"
            }

            st.subheader("🔮 Prediction Probabilities")
            for i in range(len(probs)):
                st.write(f"{label_map[i]}: {probs[i]*100:.1f}%")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
