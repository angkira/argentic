FROM eclipse-mosquitto:2.0

# Create minimal mosquitto config
RUN echo "listener 1883" > /mosquitto/config/mosquitto.conf && \
    echo "allow_anonymous true" >> /mosquitto/config/mosquitto.conf && \
    echo "persistence false" >> /mosquitto/config/mosquitto.conf && \
    echo "log_dest stdout" >> /mosquitto/config/mosquitto.conf && \
    echo "log_type all" >> /mosquitto/config/mosquitto.conf 