#!/bin/bash

# start run.sh
echo "Start run.sh"

echo $RUN_PORT  # (nothing)
echo $HOST      # (nothing)

# Set default values if env vars aren't provided
RUN_PORT=${RUN_PORT:-5000}
RUN_HOST=${HOST:-0.0.0.0}

# move to the app directory
cd /app/scripts

echo " Runnign app...."

# Run the app using Gunicorn with UvicornWorker
gunicorn -k uvicorn.workers.UvicornWorker -b $RUN_HOST:$RUN_PORT api:app

