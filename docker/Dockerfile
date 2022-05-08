FROM nvcr.io/nvidia/pytorch:20.06-py3

MAINTAINER trabit

RUN apt-get update

RUN apt-get install -y \
    gcc-4.8 \
    g++-4.8 \
    make \
    cmake \
    git

# Install MONAI
RUN python -m pip install 'monai[all]'==0.5.2 SimpleITK==2.1.0

# Build NiftyReg
RUN git clone https://github.com/KCL-BMEIS/niftyreg.git && mkdir niftyreg_build && mkdir niftyreg_install
RUN cd niftyreg_build \
    && cmake ../niftyreg \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=../niftyreg_install \
    && make \
    && make install
# Set the environment variables for NiftyReg
ARG NIFTYREG_INSTALL="/workspace/niftyreg_install"
ENV PATH="$PATH:$NIFTYREG_INSTALL/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NIFTYREG_INSTALL/lib"
RUN echo $PATH && echo $LD_LIBRARY_PATH

# Copy this repository in /feta_seg
ADD . /feta_seg
