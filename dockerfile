FROM ghcr.io/mlflow/mlflow:latest
WORKDIR /app
# install dependency
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# First create the user properly with a home directory and shell
RUN adduser --disabled-password --gecos "" --uid 1000 mlflow-user && \
    mkdir -p /app/models /app/tracking/mlflow_artifacts && \
    chown -R mlflow-user:mlflow-user /app
# Install extra dependencies
RUN apt-get update && apt-get install -y postgresql-client && \
    pip install psycopg2-binary
USER mlflow-user


COPY --chown=mlflow-user:mlflow-user run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["bash", "/app/run.sh"]