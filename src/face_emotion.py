'''
Face emotion client for intu
'''

import sys
import os
import socket
import argparse
import configparser
import uuid
import time
import math
import logging

from functools import partial
from py_watson_cloud_publisher import publish

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_emotion')

class FaceEmotionClient(object):

    def on_connected(self):
        logger.info("On Connected function!")

    @staticmethod
    def publish_emotion(emotion_code, emotion_text, emotion_probability):
        logger.info("publishing emotion with code %s and text %s", emotion_code, emotion_text)
        emotion = Thing()
        emotion.category = ThingCategory.PERCEPTION
        emotion.set_type("IThing")
        data = {'m_Text': emotion_text, 'ecode': emotion_code, 'eprob': emotion_probability,
                'time': int(round(time.time() * 1000))}
        emotion.data = data
        emotion.data_type = "FaceEmotion"
        logger.info("thing: %s", emotion.data)
        Blackboard.get_instance().add_thing(emotion, "")

    def run(self, config):

        start_time = time.time()
        end_time = start_time + float(config.get("intu", "timeout"))
        logger.info("starting time is %s", start_time)
        logger.info("end time is %s", end_time)

        while not math.isclose(end_time, time.time(), abs_tol=5):
            self_id = str(uuid.uuid4())
            headers = [('selfId', self_id), ('token', config.get("intu", "token"))]
            topic = None
            try:

                topic = TopicClient.start_instance(config.get("intu", "host"), int(config.get("intu", "port")), headers)
                TopicClient.get_instance().setHeaders(self_id, config.get("intu", "token"))
                TopicClient.get_instance().set_callback(self.on_connected)
                logger.info("trying to connect...")
                topic.reactor.connect()
                return topic

            except KeyboardInterrupt:
                exit()
            except (ConnectionRefusedError, TimeoutError) as ex:
                logger.error("connection error is %s", ex)
                logger.info("will retry in 10 seconds...")
                topic.reactor.close_connection()
                topic = None
                # sleep before the next attempt
                time.sleep(10)
                continue

        logger.info("intu connection timeout, exiting...")
        # should exit here, since it's timeouted
        exit()

def env_or_config(config, envvar, section, key):
    if envvar in os.environ:
        return os.environ[envvar]
    else:
        # let the exceptions fly if the required config fallback is not defined
        return config.get(section, key)

def main(argv):
    config_file = "face_emotion.cfg"
    config, args, cam_source = parse_config(argv, config_file)
    print_config(config, cam_source)

    eoc = partial(env_or_config, config)

    # instantiate publisher
    publisher = publish.CachePublisher(
        logging.getLogger('face_emotion.Publisher'), \
        int(eoc('FC_MQTT_PUB_INTERVAL', 'mqtt', 'pub_interval')), \
        int(eoc('FC_MQTT_RECORDS_MAX', 'mqtt', 'records_interval_max_publish')), \
        int(eoc('FC_CLOUDANT_PUB_INTERVAL', 'cloudant', 'pub_interval')), \
        int(eoc('FC_CLOUDANT_RECORDS_MAX', 'cloudant', 'records_interval_max_publish')), \
        {
            'hostname': eoc('FC_MQTT_HOSTNAME', 'mqtt', 'connection_hostname'),
            'port': int(eoc('FC_MQTT_PORT', 'mqtt', 'connection_port')),
            'client_id': eoc('FC_MQTT_CLIENT_ID', 'mqtt', 'connection_client_id'),
            'topic': eoc('FC_MQTT_TOPIC', 'mqtt', 'connection_topic'),
            'auth': {
                'username': eoc('FC_MQTT_USERNAME', 'mqtt', 'connection_auth_username'),
                'password': eoc('FC_MQTT_PASSWORD', 'mqtt', 'connection_auth_password'),
            },
            'tls': {
                'ca_certs': eoc('FC_MQTT_CACERTS_PATH', 'mqtt', 'connection_tls_ca_certs'),
            }
        }, \
        {
            'username': eoc('FC_CLOUDANT_USERNAME', 'cloudant', 'connection_username'),
            'password': eoc('FC_CLOUDANT_PASSWORD', 'cloudant', 'connection_password'),
            'url': eoc('FC_CLOUDANT_URL', 'cloudant', 'connection_url'),
            'db': eoc('FC_CLOUDANT_DB', 'cloudant', 'connection_db'),
        },
    )

    # connects to intu if the param is specified
    if args.intu:
        from self.topics.topic_client import TopicClient
        from self.blackboard.blackboard import Blackboard
        from self.blackboard.thing import Thing
        from self.blackboard.thing import ThingCategory

        fc = FaceEmotionClient()
        topic = fc.run(config)
        inference(topic, args, cam_source, config, publisher)

    else:
        inference(None, args, cam_source, config, publisher)

def inference(topic, args, cam_source, config, publisher):
    from statistics import mode
    import cv2
    from keras.models import load_model
    import numpy as np

    from utils.datasets import get_labels
    from utils.inference import detect_faces
    from utils.inference import draw_text
    from utils.inference import draw_bounding_box
    from utils.inference import apply_offsets
    from utils.inference import load_detection_model
    from utils.preprocessor import preprocess_input

    logger.info("inference")

    if args.intu:
        logger.info("inference(): results will be published to intu")
    else:
        logger.info("inference(): working standalone")

    # parameters for loading data and images
    detection_model_path = config.get("model", "detection")
    emotion_model_path = config.get("model", "emotion")
    emotion_labels = get_labels('fer2013')

    # timeout for video source access
    if args.intu:
        timeout = float(config.get("intu", "video_timeout"))
    else:
        timeout = 0
    end_time = time.time() + timeout

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    # starting video streaming
    cv2.namedWindow('emotion_inference')

    video_capture = cv2.VideoCapture(cam_source)
    while True:
        ret, bgr_image = video_capture.read()
        logger.debug("inference: video_capture.read() ret is %s", ret)
        while not ret:
            logger.info("inference: error occurred capturing video, ensure your camera is accessible to the system and you've the appropriate numeral to access it")
            logger.info("end_time is %s", str(end_time))
            logger.info("the current time is %s", str(time.time()))
            if math.isclose(end_time, time.time(), abs_tol=5.0):
                logger.info("video stream access timeout, exiting...")
                exit()
            else:
                logger.info("waiting 10 seconds for the next try, delta %s", end_time - time.time())
                time.sleep(10)
                ret, bgr_image = video_capture.read()

        end_time = time.time() + 20
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue
            logger.info("emotion is %s, with probability %s", emotion_text, emotion_probability)
            publisher.write_and_pub({'emotion_text': emotion_text, 'emotion_probability': float(emotion_probability), 'hostname': socket.gethostname()})

            if args.intu:
                FaceEmotionClient.publish_emotion(emotion_label_arg, emotion_text, emotion_probability)

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('emotion_inference', bgr_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("canceled with keyboard, exiting...")
            break
        if args.intu and (not topic.is_connected):
            logger.info("disconnected from intu, exiting...")
            break


def parse_config(argv, config_file):
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configuration', help='configuration file', default=config_file)
    parser.add_argument('-intu', help='connects to intu instance', action='store_true')
    args = parser.parse_args()

    # originally there was configuration here for any of a number of input methods; we are using camera only
    cam_source = int(os.environ.get('FC_CAMERA_SOURCE', '0'))

    config = configparser.ConfigParser()

    if args.configuration is not None:
        config_file = args.configuration

    if os.path.isfile(config_file):
        config.read(config_file)
    else:
        logger.info("Error: configuration file %s is missing or can't be read", config_file)
        sys.exit(2)

    return config, args, cam_source


def print_config(config, cam_source):
    logger.info("====================================================")
    logger.info("Running with the following configuration:")
    logger.info("====================================================")
    if 'intu' in config.sections():
        logger.info("host is " + config.get("intu", "host"))
        logger.info("port is " + config.get("intu", "port"))
        logger.info("token is " + config.get("intu", "token"))
        logger.info("connection timeout is " + config.get("intu", "timeout"), "sec")
        logger.info("video timeout is " + config.get("intu", "video_timeout"), "sec")
        logger.info("====================================================")
    logger.info("input: (camera)")
    logger.info("source: " + str(cam_source))
    logger.info("====================================================")
    logger.info("face detection model: " + config.get("model", "detection"))
    logger.info("face emotion model: " + config.get("model", "emotion"))

if __name__ == "__main__":
    main(sys.argv[1:])
