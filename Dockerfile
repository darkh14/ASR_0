# FROM nvcr.io/nvidia/nemo:23.10
FROM python:3.12.4
LABEL authors="darkh14"

RUN mkdir /models_data

ADD models_data /models_data

EXPOSE 8060:8060

RUN mkdir -p /data/source
RUN mkdir -p /data/audio
RUN mkdir -p /data/decoded_audio
RUN mkdir -p /data/result

WORKDIR /app

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install Cython
RUN pip install git+https://github.com/NVIDIA/NeMo.git@1fa961ba03ab5f8c91b278640e29807079373372#egg=nemo_toolkit[asr]
RUN pip install -U soundfile
RUN pip install pyannote.audio==3.2.0

RUN pip uninstall huggingface_hub -y
RUN pip install huggingface-hub==0.22

COPY ./app .

RUN python3 -m pip install --root-user-action=ignore --upgrade pip && python3 -m  pip install --root-user-action=ignore -r requirements.txt

ENTRYPOINT ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8060"]
