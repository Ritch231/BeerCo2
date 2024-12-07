import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# 加载模型和多项式特征转换器
model = joblib.load('pressure_model.pkl')
poly = joblib.load('poly_transformer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temperature = data.get('temperature')
    co2_concentration = data.get('co2_concentration')

    if temperature is None or co2_concentration is None:
        return jsonify({'error': 'Temperature and CO2 concentration are required'}), 400

    # 输入数据并进行预测
    input_data = np.array([[temperature, co2_concentration]])
    input_poly = poly.transform(input_data)
    predicted_pressure = model.predict(input_poly)

    return jsonify({'predicted_pressure': predicted_pressure[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=606)