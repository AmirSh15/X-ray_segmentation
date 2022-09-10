FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# set shell bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

#install anaconda
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# set conda environment
COPY environment.yml /tmp/environment.yml
RUN conda update && conda env create --name segmentation --file /tmp/environment.yml

RUN echo "conda activate segmentation" >> ~/.bashrc
ENV PATH /opt/conda/envs/segmentation/bin:$PATH
ENV CONDA_DEFAULT_ENV segmentation

#set WORKDIR
WORKDIR /code

COPY segmentation/train.py /tmp/train.py
ENTRYPOINT ["python", "/tmp/train.py"]