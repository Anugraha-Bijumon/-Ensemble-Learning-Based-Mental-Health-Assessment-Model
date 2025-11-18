from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load models and label encoder
bagging_model = joblib.load('models/bagging_model.pkl')
boosting_model = joblib.load('models/boosting_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

app = Flask(__name__)

# Route to display the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data, excluding 'Patient_Number'
        input_data = [float(request.form[col]) for col in request.form if col != 'Patient_Number']
        input_data = np.array(input_data).reshape(1, -1)

        # Get predictions from each model
        bagging_pred = bagging_model.predict(input_data)
        boosting_pred = boosting_model.predict(input_data)
        rf_pred = rf_model.predict(input_data)

        # Convert predictions back to original labels
        bagging_result = label_encoder.inverse_transform(bagging_pred)[0]
        boosting_result = label_encoder.inverse_transform(boosting_pred)[0]
        rf_result = label_encoder.inverse_transform(rf_pred)[0]

        # Prepare result dictionary
        results = {
            "Bagging Model Prediction": bagging_result,
            "Boosting Model Prediction": boosting_result,
            "Random Forest Model Prediction": rf_result
        }

        return render_template('index.html', prediction_results=results)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
