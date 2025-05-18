FROM python:3.11.12

WORKDIR /work
COPY requirements.txt /work

RUN pip install -r requirements.txt