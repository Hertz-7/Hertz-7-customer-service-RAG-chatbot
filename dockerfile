# Use the NVIDIA CUDA image with cuDNN support
FROM nvidia/cuda:12.4.1-cudnn-devel-rockylinux8

# Set the working directory
WORKDIR /app

# Install dependencies and Python
RUN yum update -y && \
    yum install -y \
    python3 \
    python3-pip \
    wget \
    unzip \
    && yum clean all

COPY . /app
# Install Python dependencies

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Install ngrok
RUN wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip \
    && unzip ngrok-stable-linux-amd64.zip \
    && mv ngrok /usr/local/bin/ngrok \
    && rm ngrok-stable-linux-amd64.zip

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application
CMD ["python3", "main.py"]
