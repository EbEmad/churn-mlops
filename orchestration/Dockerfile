FROM apache/airflow:2.9.1-python3.10

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh /run.sh


CMD ["bash","/run.sh"]