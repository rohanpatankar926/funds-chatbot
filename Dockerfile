FROM python:3.11-slim
WORKDIR /app
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
