# Washing Machine Notifier

The my washing machine doesn't have an alarm to signify that it's done, but I had a spare webcam and a few hours of time so I made something better.

### Environment Variables

The following environment variables need to be configured:

- `MQTT_BROKER`: MQTT broker hostname/IP address
- `MQTT_PORT`: MQTT broker port (default: 1883)
- `MQTT_TOPIC`: MQTT topic for publishing status (default: homeassistant/washingmachine/status)
- `MQTT_USERNAME`: MQTT username (optional)
- `MQTT_PASSWORD`: MQTT password (optional)
- `RTSP_URL`: URL to the RTSP stream for monitoring the washing machine

### Running with Podman

```bash
podman run -d --name washing-machine-notifier \
  --restart unless-stopped \
  -e MQTT_BROKER=your-mqtt-broker \
  -e MQTT_PORT=1883 \
  -e MQTT_TOPIC=home/washingmachine/status \
  -e MQTT_USERNAME=your-username \
  -e MQTT_PASSWORD=your-password \
  -e RTSP_URL=rtsp://your-camera-ip:port/path \
  ghcr.io/braccae/washing-machine-notifier:latest
```
Also included is a quadlet file for systemd.

## Local Development

For local development and testing:

```bash
# Clone the repository
git clone https://github.com/your-username/washing-machine-notifier.git
cd washing-machine-notifier

# Create and configure .env file
cp example.env .env
# Edit .env with your settings

# Install dependencies
pip install -r requirements.txt

# Run the application
python washer_monitor.py
```

## Calibration

For initial setup and calibration:

```bash
python calibrate.py
```

Follow the on-screen instructions to identify the indicator lights on your washing machine.
