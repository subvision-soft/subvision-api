# Use a compatible base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /

# Copy the requirements file
COPY ./requirements.txt /requirements.txt

# Install system dependencies and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /requirements.txt \
    && apt-get remove -y build-essential && apt-get autoremove -y && apt-get clean

# Copy the rest of the application
COPY ./ /

# Expose port 443
EXPOSE 443

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443"]
