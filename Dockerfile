FROM python:3.6

LABEL Name=watershed Version=0.0.1

WORKDIR /app
ADD . /app

RUN python3 -m pip install -r requirements.txt
CMD ["python3", "./watershed.py"]
