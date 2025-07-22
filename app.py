from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load(open("gwp.pkl", "rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/submit', methods=["POST"])
def submit():
    if request.method == "POST":
        try:
            # Retrieve form inputs
            incentive = float(request.form['incentive'])
            over_time = float(request.form['over_time'])
            no_of_style_change = float(request.form['no_of_style_change'])
            smv = float(request.form['smv'])
            month = float(request.form['month'])
            weekday = float(request.form['weekday'])
            department_sewing = float(request.form['department_sewing'])  # 1 if sewing, else 0

            input_data = np.array([[incentive, over_time, no_of_style_change, smv, month, weekday, department_sewing]])

            prediction = model.predict(input_data)[0]
            prediction = round(prediction, 2)

            return render_template("submit.html", prediction=prediction)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("predict.html")


if __name__ == '__main__':
    app.run(debug=True)
