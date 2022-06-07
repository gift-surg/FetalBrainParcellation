# @brief  Docker image to install the environment to run the code in this repository.
# @author Lucas Fidon (lucas.fidon@gmail.com).
# @date   June 2022.

# Image info
FROM nvcr.io/nvidia/pytorch:21.08-py3
LABEL maintainer="lucas.fidon@gmail.com"
LABEL version="0.1"
LABEL description="Docker for trustworthy fetal brain 3D T2w MRI segmentation."

# Add third-party softwares
ADD docker/third-party FetalBrainParcellation/third-party

# Install commonly used packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    aptitude sudo vim zsh wget curl git net-tools \
    rxvt-unicode tmux thunar locales build-essential cmake \
    python3-pip
RUN echo "alias python=python3" >> ~/.bash_aliases

# Remove unnecesary packages
RUN DEBIAN_FRONTEND=noninteractive apt remove -y --purge gdm3 xserver-xephyr

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install MONAI (used for deep learning)
RUN python -m pip install 'monai[all]'==0.7.0 SimpleITK==2.1.0 tensorboardX pyyaml nested_dict scipy==1.6.3


# Copy necessary files
COPY src FetalBrainParcellation/src
COPY data FetalBrainParcellation/data
COPY run_segment.py FetalBrainParcellation/run_segment.py

# NiftyReg (used for medical image registration)
ARG NIFTYREG="/workspace/FetalBrainParcellation/third-party/niftyreg/build/reg-apps"
ENV PATH="$PATH:$NIFTYREG"

# RUN CODE - the user can pass the argument of the run_segment.py script when creating a docker container
ENTRYPOINT ["python","/workspace/FetalBrainParcellation/run_segment.py"]
