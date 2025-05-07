#!/bin/bash

if [ -n "$RUN_PORT" ]; then
    echo "RUN_PORT is set to $RUN_PORT"
else
    echo "RUN_PORT is not set"
fi

if [ -n "$HOST" ]; then
    echo "HOST is set to $HOST"
else
    echo "HOST is not set"
fi

# Set default values if env vars aren't provided
RUN_PORT=${RUN_PORT:-5000}
RUN_HOST=${HOST:-0.0.0.0}

# move to the app directory
cd /app/scripts

echo " Runnign app...."
# Run the app using Gunicorn with UvicornWorker
# Run the app using Gunicorn with UvicornWorker in the background
gunicorn -k uvicorn.workers.UvicornWorker -b $RUN_HOST:$RUN_PORT api:app &

# Wait a second or two (optional, but helps avoid race conditions)
cd ..
sleep 2

# Print that we're starting
echo "Starting Jupyter Notebook..."

# Run Jupyter Notebook
jupyter notebook \
  --ip=0.0.0.0 \
  --port=8888 \
  --allow-root \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password=''
