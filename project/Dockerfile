FROM python:3.11.4

WORKDIR /project
COPY . .
RUN apk add --update --no-cache --virtual .tmp-build-deps \
    gcc libc-dev linux-headers && pip install --no-cache-dir -r requirements.txt