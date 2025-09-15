FROM public.ecr.aws/pytorch/pytorch:2.0.0-cpu-py3
RUN pip install ultralytics clip Pillow numpy
COPY inference.py /opt/program/inference.py
COPY YOLOs.pt /opt/ml/model/YOLOs.pt
ENV SAGEMAKER_PROGRAM inference.py