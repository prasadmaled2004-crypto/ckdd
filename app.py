from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoders
model = joblib.load('models/ckd_model.pkl')
le_dict = joblib.load('models/label_encoders.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Columns as per preprocessed_ckd.csv
    input_cols = [
        'age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot',
        'hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'
    ]
    input_data = []
    for col in input_cols:
        value = request.form[col]
        # Encode categorical features
        if col in le_dict:
            # Only encode if value is in encoder's classes
            if value in le_dict[col].classes_:
                value = le_dict[col].transform([value])[0]
            else:
                # Handle unseen label: fallback or error
                value = le_dict[col].transform([le_dict[col].classes_[0]])[0]
        else:
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass
        input_data.append(value)
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    pred_label = le_dict['prediction'].inverse_transform([prediction])[0]
    result = "CKD Detected" if pred_label == 'ckd' else "No CKD"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)