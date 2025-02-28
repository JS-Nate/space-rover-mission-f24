# Use Python 3.8 as the base image
FROM python:3.8

# Set environment variables to prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VENV_DIR="/app/venv"
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV MKL_NUM_THREADS=4


# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    xvfb \
    x11-xserver-utils \
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements into the container
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN if [ ! -d ${PYTHON_VENV_DIR} ]; then \
        echo "Creating venv for space rover gesture control service in ${PYTHON_VENV_DIR}"; \
        python3 -m venv ${PYTHON_VENV_DIR}; \
    fi && \
    . ${PYTHON_VENV_DIR}/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code (including Python files and best.pt) into the container
COPY . .

# Run the script when the container starts (WITHOUT xvfb-run)
CMD ["/bin/bash", "-c", "source ${PYTHON_VENV_DIR}/bin/activate && python3 ChooseDriving.py"]
