# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# import eventlet
# import eventlet.wsgi
# import socketio
# import numpy as np
# import base64
# import cv2
# from flask import Flask
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError
# from io import BytesIO
# from PIL import Image

# # Create a Socket.IO server instance
# sio = socketio.Server(logger=True, engineio_logger=True)
# app = Flask(__name__)

# # Define constants
# MAX_SPEED = 10

# # Load the model once at the start
# def load_trained_model():
#     mse = MeanSquaredError()
#     model = load_model('model.h5', custom_objects={'mse': mse})
#     print('Model loaded successfully')
#     return model

# model = load_trained_model()

# # Image preprocessing function
# def preprocess_image(img):
#     img = img[60:135, :, :]  # Crop the image
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
#     img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
#     img = cv2.resize(img, (200, 66))  # Resize the image
#     img = img / 255.0  # Normalize the image
#     return img

# # Handle telemetry data
# @sio.on('telemetry')
# def telemetry(sid, data):
#     if data:
#         try:
#             speed = float(data['speed'])
#             image_data = data['image']
#             image = Image.open(BytesIO(base64.b64decode(image_data)))
#             image = np.asarray(image)
#             image = preprocess_image(image)
#             image = np.array([image])
#             steering = float(model.predict(image))
#             throttle = 1.0 - speed / MAX_SPEED
#             send_control(steering, throttle)
#             print(f'Steering: {steering:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}')
#         except Exception as e:
#             print(f'Error processing telemetry: {e}')
#     else:
#         print('No telemetry data received')

# # Handle new connections
# @sio.on('connect')
# def connect(sid, environ):
#     print(f'Client connected: {sid}')
#     send_control(0, 0)

# # Handle disconnections
# @sio.on('disconnect')
# def disconnect(sid):
#     print(f'Client disconnected: {sid}')

# # Send control commands to the simulator
# def send_control(steering, throttle):
#     try:
#         sio.emit('steer', data={
#             'steering_angle': str(steering),
#             'throttle': str(throttle)
#         })
#     except Exception as e:
#         print(f'Error sending control: {e}')

# # Main entry point
# if __name__ == '__main__':
#     try:
#         app = socketio.WSGIApp(sio, app)
#         eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
#     except Exception as e:
#         print(f'Error during server setup: {e}')




import os
import eventlet
import eventlet.wsgi
import socketio
import numpy as np
import base64
import cv2
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from io import BytesIO
from PIL import Image
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a Socket.IO server instance
sio = socketio.Server(logger=True, engineio_logger=True)
app = Flask(__name__)

# Define constants
MAX_SPEED = 10

# Load the model once at the start
def load_trained_model():
    mse = MeanSquaredError()
    model = load_model('model.h5', custom_objects={'mse': mse})
    logger.info('Model loaded successfully')
    return model

model = load_trained_model()

# Image preprocessing function
def preprocess_image(img):
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize the image
    img = img / 255.0  # Normalize the image
    return img

# Handle telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            speed = float(data['speed'])
            image_data = data['image']
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = np.asarray(image)
            image = preprocess_image(image)
            image = np.array([image])
            steering = float(model.predict(image))
            throttle = 1.0 - speed / MAX_SPEED
            send_control(steering, throttle)
            logger.info(f'Steering: {steering:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}')
        except Exception as e:
            logger.error(f'Error processing telemetry: {e}')
    else:
        logger.warning('No telemetry data received')

# Handle new connections
@sio.on('connect')
def connect(sid, environ):
    logger.info(f'Client connected: {sid}')
    send_control(0, 0)

# Handle disconnections
@sio.on('disconnect')
def disconnect(sid):
    logger.info(f'Client disconnected: {sid}')

# Send control commands to the simulator
def send_control(steering, throttle):
    try:
        sio.emit('steer', data={
            'steering_angle': str(steering),
            'throttle': str(throttle)
        })
    except Exception as e:
        logger.error(f'Error sending control: {e}')

# Main entry point
if __name__ == '__main__':
    try:
        app = socketio.WSGIApp(sio, app)
        eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 4567)), app)
    except Exception as e:
        logger.critical(f'Error during server setup: {e}')
