# face_classification intu client

## Description

The client detects face emotions during inference and sends the information to the running intu instance.

## Run the client

from the <project/src> directory launch the command

`python3 face_emotion.py --intu`

## Client configuration

The client is configured with the `face_emotion.cfg` file:

```
[intu]
host = 127.0.0.1
```

### Data

The emotion data sending in the following JSON format:

'm_Text': emotion_text,
'ecode': emotion_code,
'time’: timestamp

The data type is `FaceEmotion`

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