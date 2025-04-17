FROM alpine:3.21


RUN apk add \
  py3-opencv \
  py3-numpy \
  py3-paho-mqtt \
  py3-dotenv


WORKDIR /app

COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "washer_monitor.py"]
