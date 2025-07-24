from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        values = [
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            int(request.form['Gender_Male']),
            int(request.form['Married_Yes']),
            int(request.form['Dependents_1']),
            0,  # Dependents_2 removed
            0,  # Dependents_3+ removed
            int(request.form['Education_Not_Graduate']),
            int(request.form['Self_Employed_Yes']),
            int(request.form['Property_Area_Semiurban']),
            int(request.form['Property_Area_Urban'])
        ]

        scaled_input = scaler.transform([values])
        prediction = model.predict(scaled_input)[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result, show_result=True)

    return render_template("index.html", show_result=False)
if __name__ == "__main__":
    app.run(debug=True)