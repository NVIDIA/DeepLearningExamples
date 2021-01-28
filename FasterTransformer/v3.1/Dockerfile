FROM nvcr.io/nvidia/pytorch:20.07-py3

RUN apt update
RUN apt install -y openssh-server
RUN apt-get install -y emacs

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PermitRootLogin/PermitRootLogin/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN echo "export VISIBLE=now" >> /etc/profile

# RUN echo "service ssh restart" >> /
# # Add user ubuntu
# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntu -p "$(openssl passwd -1 ubuntu)"

# restart ssh service
ENTRYPOINT service ssh restart && /bin/bash