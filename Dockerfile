FROM openhorizon/aarch64-tx2-face-classification-intu:JetPack3.2-RC

RUN apt-get update && apt-get install -y --no-install-recommends \
				xvfb \
				xauth \
				x11vnc \
				x11-utils \
				x11-xserver-utils \
				x11-apps

# start x11vnc and expose its port
ENV DISPLAY :0.0
EXPOSE 5900
COPY entrypoint-x11.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix && chown root /tmp/.X11-unix

ENTRYPOINT ["/entrypoint.sh"]
