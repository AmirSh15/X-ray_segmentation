FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update \
    && apt-get -y install build-essential ffmpeg libsm6 libxext6 wget nano git \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_ALWAYS_YES=true

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old

# install detectron2
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# install requirements
COPY . /X-ray_segmentation
RUN pip install -r /X-ray_segmentation/requirements.txt

# install spacy
RUN conda install spacy
RUN pip install markupsafe==2.0.1
RUN python -m spacy download en_core_web_sm

WORKDIR /X-ray_segmentation
ENV PYTHONPATH "${PYTHONPATH}:/X-ray_segmentation"

