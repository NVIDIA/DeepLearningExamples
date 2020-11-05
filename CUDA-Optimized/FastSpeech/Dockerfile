ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.10-py3
FROM ${FROM_IMAGE_NAME}

# ARG UNAME
# ARG UID
# ARG GID
# RUN groupadd -g $GID -o $UNAME
# RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
# USER $UNAME

ADD . /workspace/fastspeech
WORKDIR /workspace/fastspeech

RUN sh ./scripts/install.sh
