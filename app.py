import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# 数据定义：温度、压力、CO2值
data = {
    "温度": [12.6, 8.8, 8.8, 7.2, 9.0, 12.6, 11.0, 7.4, 5.5, 9.0, 6.1, 7.5, 5.7, 1.8, 6.0, 1.9, 2.7, 6.1, 2.9, 4.0,
           4.4, 8.6, 8.8, 11.0, 5.2, 5.5, 7.7, 8.9, 12.6, 7.7, 3.2, 3.7, 3.0, 4.4, 2.9, 4.0, 2.8, 4.2, 3.9, 4.0,
           3.6, 3.6, 3.9, 2.4, 4.0, 4.1, 8.7, 4.4, 8.5, 2.7],
    "压力": [0.51, 0.35, 0.38, 0.39, 0.49, 0.70, 0.66, 0.50, 0.42, 0.59, 0.45, 0.52, 0.44, 0.26, 0.46, 0.27, 0.30, 0.46,
           0.32, 0.35, 0.39, 0.60, 0.61, 0.73, 0.43, 0.45, 0.56, 0.62, 0.90, 0.57, 0.35, 0.46, 0.51, 0.69, 0.54, 0.61,
           0.55, 0.65, 0.66, 0.67, 0.66, 0.66, 0.68, 0.60, 0.69, 0.70, 1.05, 0.90, 1.19, 0.86],
    "CO2值": [0.313, 0.318, 0.330, 0.345, 0.349, 0.353, 0.364, 0.370, 0.372, 0.372, 0.373, 0.374, 0.375, 0.376, 0.376,
           0.377, 0.377, 0.377, 0.378, 0.378, 0.378, 0.378, 0.378, 0.378, 0.379, 0.380, 0.380, 0.380, 0.380, 0.382,
           0.383, 0.407, 0.432, 0.435, 0.440, 0.444, 0.445, 0.453, 0.458, 0.461, 0.464, 0.465, 0.465, 0.466, 0.466,
           0.468, 0.480, 0.516, 0.518, 0.538],
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 特征处理：温度和压力为输入特征，CO2为目标值
X = df[["温度", "压力"]]
y = df["CO2值"]

# 多项式回归模型（2阶）
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 模型拟合
model = LinearRegression()
model.fit(X_poly, y)

@app.route('/predict', methods=['POST'])
def predict():
    # 获取请求数据
    data = request.get_json()
    temperature = data.get('temperature')
    co2_concentration = data.get('co2_concentration')

    if temperature is None or co2_concentration is None:
        return jsonify({'error': 'Temperature and CO2 concentration are required'}), 400

    # 输入数据并进行多项式特征转换
    input_data = np.array([[temperature, co2_concentration]])
    input_poly = poly.transform(input_data)

    # 使用模型进行预测
    predicted_co2 = model.predict(input_poly)[0]

    # 使用预测的 CO2 值计算压力
    predicted_pressure = None
    for pressure in np.arange(0.2, 1.51, 0.01):
        # 反向转换 CO2 和压力的关系
        test_input = np.array([[temperature, pressure]])
        test_input_poly = poly.transform(test_input)
        predicted_co2_test = model.predict(test_input_poly)[0]

        if np.abs(predicted_co2_test - predicted_co2) < 0.001:  # 阈值判断
            predicted_pressure = pressure
            break

    if predicted_pressure is not None:
        return jsonify({'predicted_pressure': round(predicted_pressure, 3)})
    else:
        return jsonify({'error': 'Could not calculate pressure for the given values'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=606)