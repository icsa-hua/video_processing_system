FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    curl \
    libglib2.0-0 \
    net-tools \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y    

# Install any python packages you need
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install opencv-python-headless
# Install PyTorch and torchvision
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
WORKDIR /app 
COPY . /app 

# Expose the Streamlit port 
EXPOSE 8000 8503

# Set the command to run when the container starts
CMD ["python3", "/app/obs_pipeline.py", "--gui", "--verbose", "--host_server=0.0.0.0"]