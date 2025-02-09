# Use Python 3.7 as the base image
FROM python:3.7-slim

# Set environment variables to prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VENV_DIR="/app/venv"

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements into the container
COPY gestures/openCV_implementation/src/requirements.txt .

# Create a virtual environment and install dependencies
RUN if [ ! -d ${PYTHON_VENV_DIR} ]; then \
        echo "Creating venv for space rover gesture control service in ${PYTHON_VENV_DIR}"; \
        python3.7 -m venv ${PYTHON_VENV_DIR}; \
    fi && \
    . ${PYTHON_VENV_DIR}/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code (optional)
COPY . .

# Activate the virtual environment when the container starts
CMD ["/bin/bash", "-c", "source ${PYTHON_VENV_DIR}/bin/activate && python your_app.py"]
