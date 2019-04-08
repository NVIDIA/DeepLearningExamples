FROM nvcr.io/nvidia/pytorch:19.03-py3

# Set working directory
WORKDIR /mlperf

RUN apt-get update && apt-get install -y python3-tk python-pip git tmux htop tree

# Necessary pip packages
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python3 -m pip install pycocotools==2.0.0

# Copy SSD code
COPY ./setup.py .
COPY ./csrc ./csrc
RUN pip install .

COPY . .
