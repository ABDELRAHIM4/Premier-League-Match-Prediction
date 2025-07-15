from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load model + team encoders
model = joblib.load('model.pkl')
le_home = joblib.load('le_home.pkl')
le_away = joblib.load('le_away.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    teams = list(le_home.keys())

    if request.method == 'POST':
        try:
            home_team = request.form['home']
            away_team = request.form['away']
            date = pd.to_datetime(request.form['date'])
            wk = int(request.form['wk'])

            

            if home_team not in le_home or away_team not in le_away:
                prediction = 'Team not found in training data'
            else:
                home_code = le_home[home_team]
                away_code = le_away[away_team]

                X = [[home_code, 0, 0, away_code, wk]]

                pred = model.predict(X)[0]

                probs = model.predict_proba(X)[0]
                label_map = {0: f'{away_team} Win', 1: 'Draw', 2: f'{home_team} Win'}

                prediction = {
                label_map[i]: f"{probs[i]*100:.1f}%" for i in range(len(probs))
                }


        except Exception as e:
            prediction = f'Error: {e}'

    
    return render_template('index.html', prediction=prediction, teams=teams)
if __name__ == '__main__':
    app.run(debug=True)
