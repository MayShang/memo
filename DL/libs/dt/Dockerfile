FROM python:3.6-alpine
WORKDIR /app
COPY . /app

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python3", "app.py"]
