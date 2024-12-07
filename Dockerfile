# 使用 Python 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制当前文件夹的内容到容器中
COPY . .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口 606
EXPOSE 606

# 启动 Flask 应用
CMD ["python", "app.py"]