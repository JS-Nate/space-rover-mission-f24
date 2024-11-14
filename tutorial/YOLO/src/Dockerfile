# Use Python 3.8 as the base image
FROM python:3.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    xvfb \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first for caching pip install
COPY requirements.txt /app/

# Install Python dependencies globally
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Default command to run the Python script
CMD ["python", "combined1.py"]
