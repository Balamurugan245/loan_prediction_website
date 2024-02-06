import pickle
from flask import Flask, render_template, request
import os
import joblib  # Import joblib directly

app = Flask(__name__)

# Define the path to your model file
model_filename = 'rf_regressor.pickle'
model_path = os.path.abspath(model_filename)

# Load the machine learning model
with open(model_path, 'rb') as pkl:
    rf_regressor = joblib.load(pkl)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get input values from the form
        education = 1 if request.form['education'] == 'Yes' else 0
        self_employed = 1 if request.form['self_employed'] == 'Yes' else 0
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        cibil_score = float(request.form['cibil_score'])

        # Make prediction using the machine learning model
        prediction = round(rf_regressor.predict([[education, self_employed, income, loan_amount, cibil_score]])[0])

    # Render the result in the HTML template
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
