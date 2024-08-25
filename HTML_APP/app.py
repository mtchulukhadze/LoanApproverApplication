from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

loaded_model = load_model(r"D:\Data\AI & ML\loan_approval_model.h5")
scaler = joblib.load(r"D:\Data\AI & ML\scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.form
        gender = int(data.get('gender'))
        married = int(data.get('married'))
        dependents = int(data.get('dependents'))
        education = int(data.get('education'))
        self_employed = int(data.get('self_employed'))
        applicant_income = float(data.get('applicant_income'))
        coapplicant_income = float(data.get('coapplicant_income'))
        loan_amount = float(data.get('loan_amount'))
        loan_amount_term = float(data.get('loan_amount_term'))
        credit_history = int(data.get('credit_history'))
        property_area = int(data.get('property_area'))

        
        new_data = np.array([[gender, married, dependents, education, self_employed,
                              applicant_income, coapplicant_income, loan_amount,
                              loan_amount_term, credit_history, property_area]])

        
        new_data_scaled = scaler.transform(new_data)

        
        prediction = loaded_model.predict(new_data_scaled)
        prediction_binary = (prediction > 0.5).astype(int)

        probability = float(prediction[0][0])
        result_class = int((prediction> 0.5).astype(int)[0][0])
        result = {
            'probability': round(probability, 2),
            'result': 'Approved' if result_class == 1 else 'Rejected'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
