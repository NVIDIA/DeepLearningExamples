FROM nvcr.io/nvidia/pytorch:19.08-py3

# Set working directory
WORKDIR /workspace

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk python-pip git tmux htop tree

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
