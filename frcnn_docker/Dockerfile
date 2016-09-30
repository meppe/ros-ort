FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

MAINTAINER Manfred Eppe manfred@eppe.eu

# setup environment
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

RUN apt-get update

# Python pip
RUN apt-get install -y python-pip

# misc
RUN apt-get install -y \
	sudo \
	nano

###############
## ROS stuff ##
###############

ENV ROS_DISTRO kinetic

RUN apt-get install -y \
	ros-kinetic-ros-base

RUN apt-get install -y \
	python-rosinstall 

# bootstrap rosdep
RUN rosdep init \
    && rosdep update


########################################################
## SSH server stuff for remote debugging with pycharm ##
########################################################
LABEL eu.indigo-datacloud.type="linux"
LABEL eu.indigo-datacloud.distribution="ubuntu"
LABEL eu.indigo-datacloud.version="16.04"
LABEL eu.indigo-datacloud.architecture="amd64"

#Install INDIGO-DataCloud's package repository (to install a specific version of Ansible)
RUN apt-get update && apt-get install -y wget
RUN wget http://repo.indigo-datacloud.eu/repos/1/indigo1-ubuntu14_04.list -O /etc/apt/sources.list.d/indigo1-ubuntu14_04.list
RUN wget -q -O - http://repo.indigo-datacloud.eu/repository/RPM-GPG-KEY-indigodc | apt-key add -

# Pre-install packages to speed IM contextualization
RUN apt-get update && \
    apt-get install -y \
        ansible \
        cloud-init \
        openssh-client \
        openssh-server \
        sshpass \
        unzip 

RUN apt-get install -y net-tools

RUN mkdir /var/run/sshd

RUN echo 'root:screencast' | chpasswd

RUN sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN wget https://github.com/OpenNebula/addon-context-linux/releases/download/v4.14.3/one-context_4.14.3.deb && dpkg -i one-context_4.14.3.deb

RUN rm -f /etc/one-context.d/00-network; rm -f /etc/one-context.d/01-dns; rm -f /etc/one-context.d/05-hostname; rm -f /etc/one-context.d/06-gen-env; rm -f /etc/one-context.d/07-grow-rootfs

COPY one-contextd /usr/sbin/one-contextd

EXPOSE 22

# Clean up
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

COPY frcnn_entrypoint.sh /frcnn_entrypoint.sh
RUN chmod +x /frcnn_entrypoint.sh

ENTRYPOINT ["/frcnn_entrypoint.sh"]
