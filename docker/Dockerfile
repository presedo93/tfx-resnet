FROM python:3.9-slim-buster

RUN useradd -ms /bin/bash scientist

USER scientist

WORKDIR /home/scientist/resnet-api

ENV PATH="/home/scientist/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

CMD [ "python", "-m", "cli", "--source", "cat.jpg", "--target", "285"]
