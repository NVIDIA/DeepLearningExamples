ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.08-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update && apt-get install -y python3

# Here's a good place to install pip reqs from JoC repo.
# At the same step, also install TRT pip reqs
WORKDIR /tmp/pipReqs
COPY requirements.txt /tmp/pipReqs/jocRequirements.txt
COPY tensorrt/requirements.txt /tmp/pipReqs/trtRequirements.txt
RUN pip install --disable-pip-version-check -U -r jocRequirements.txt -r trtRequirements.txt


WORKDIR /workspace/jasper
COPY . .

