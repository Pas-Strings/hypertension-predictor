from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

print("Loading models...")
model1 = joblib.load('model1_binary.pkl')
model2 = joblib.load('model2_stage.pkl')
scaler1 = joblib.load('scaler1.pkl')
scaler2 = joblib.load('scaler2.pkl')
features = joblib.load('features.pkl')
label_encoder = joblib.load('label_encoder.pkl')
stage_display = joblib.load('stage_display.pkl')
print("Models loaded!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    systolic_bp = float(request.form['systolic_bp'])
    diastolic_bp = float(request.form['diastolic_bp'])
    cholesterol = float(request.form['cholesterol'])
    
    input_data = np.array([[age, bmi, systolic_bp, diastolic_bp, cholesterol]])
    input_scaled = scaler1.transform(input_data)
    
    prob = model1.predict_proba(input_scaled)[0]
    is_hypertensive = model1.predict(input_scaled)[0]
    
    result = {
        'hypertensive': bool(is_hypertensive),
        'prob_normal': round(prob[0] * 100, 1),
        'prob_hypertensive': round(prob[1] * 100, 1),
        'stage': None,
        'stage_display': None,
        'message': None
    }
    
    if is_hypertensive == 1:
        input_scaled2 = scaler2.transform(input_data)
        stage_idx = model2.predict(input_scaled2)[0]
        stage_code = label_encoder.inverse_transform([stage_idx])[0]
        result['stage'] = stage_code
        result['stage_display'] = stage_display.get(stage_code, stage_code)
        result['message'] = f"⚠️ You may have {stage_display.get(stage_code, stage_code)}. Please consult a doctor."
    else:
        result['message'] = "✅ Your blood pressure appears normal. Maintain a healthy lifestyle!"
    
    return render_template('result.html', result=result, input_data=request.form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)