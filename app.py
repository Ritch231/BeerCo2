import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)
CORS(app)  # 启用 CORS，允许跨域请求

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

# 构造网格化温度和压力
温度范围 = np.arange(1.0, 12.7, 0.1)  # 从1.0递增到12.7，步长0.1
压力范围 = np.arange(0.2, 1.51, 0.01)  # 从0.2递增到1.51，步长0.01

# 创建网格数据
grid_温度, grid_压力 = np.meshgrid(温度范围, 压力范围)
grid_input = np.c_[grid_温度.ravel(), grid_压力.ravel()]
grid_poly = poly.transform(grid_input)

# 预测 CO2 值
predicted_CO2 = model.predict(grid_poly).reshape(grid_压力.shape)

@app.route('/predict', methods=['POST'])
def predict():
    # 获取请求数据
    data = request.get_json()
    temperature = data.get('temperature')
    co2_concentration = data.get('co2_concentration')

    if temperature is None or co2_concentration is None:
        return jsonify({'error': 'Temperature and CO2 concentration are required'}), 400

    # 在网格中查找最接近的 CO2 值
    min_distance = float('inf')
    predicted_pressure = None

    for i in range(grid_温度.shape[0]):
        for j in range(grid_温度.shape[1]):
            grid_temperature = grid_温度[i, j]
            grid_pressure = grid_压力[i, j]
            grid_co2 = predicted_CO2[i, j]

            # 计算温度和 CO2 值的差距
            distance = np.sqrt((grid_temperature - temperature) ** 2 + (grid_co2 - co2_concentration) ** 2)

            if distance < min_distance:
                min_distance = distance
                predicted_pressure = grid_pressure

    if predicted_pressure is not None:
        return jsonify({'predicted_pressure': round(predicted_pressure, 3)})  # 返回的压力值保留小数点后3位
    else:
        return jsonify({'error': 'Could not calculate pressure for the given values'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=606)