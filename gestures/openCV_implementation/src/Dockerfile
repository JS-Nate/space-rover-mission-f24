# Use a CUDA-enabled base image with Python
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Set environment variables to prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VENV_DIR="/app/venv"
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    xvfb \
    x11-xserver-utils \
    ffmpeg \
    python3.8 \
    python3-pip \
    python3.8-venv \
    tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python3.8 -m venv ${PYTHON_VENV_DIR} && \
    . ${PYTHON_VENV_DIR}/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Ensure NVIDIA libraries are accessible
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run the script when the container starts
CMD ["/bin/bash", "-c", "source ${PYTHON_VENV_DIR}/bin/activate && python3 ChooseDriving.py"]
