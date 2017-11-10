# face_classification intu client

## Description

The client detects face emotions during inference and sends the information to the running intu instance.

## Run the client

from the <project/src> directory launch the command

`python3 face_emotion.py [-intu] <input> <source>`

intu - connect and publish inference results to intu

<input> - inference input, can be `camera`, `file`

<source> - camera index for `camera` or file name for `file`

1. Standalone with camera as input

`python3 face_emotion.py camera 0`

2. Standalone with a video file

`python3 face_emotion.py file "1.mp4"`

3. Publishing inference results from file to intu

`python3 face_emotion.py -intu file "1.mp4"`

## Client configuration

The client is configured with the `face_emotion.cfg` file:

```
[intu]
host = 127.0.0.1
port = 9443
token =

[model]

detection = ../trained_models/detection_models/haarcascade_frontalface_default.xml
emotion = ../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5
```

### Data

The emotion data sending in the following JSON format:

'm_Text': emotion_text,
'ecode': emotion_code,
'eprob': emotion_probability,
'time’: timestamp

The data type published to intu is `FaceEmotion`

### Receiving data from intu

In order to receive the data you need to subscribe to the `FaceEmotion` data type on the blackboard.

The Python code snipped looks like:

1. Subscription
```
Blackboard.get_instance().subscribe_to_type("FaceEmotion", ThingEventType.ADDED, "", self.on_emotion)
```

`self.on_emotion` is a callback function (usually it’s a method of your agent class derived from the `Agent` class) to be called when a new thing of the `FaceEmotion` data type was added to the blackboard.

2. Call back function:

```
def on_emotion(self, payload):
        ''' Handle a received Text object '''
        for key, value in payload['thing'].items():
            print (key,value)
        print("FaceEmotionAgent OnText(): " + payload['thing']['m_Data']['m_Text’])
        
```