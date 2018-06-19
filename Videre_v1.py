import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import time
import zipfile
import serial
import time
import os

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from twilio.rest import Client
import cv2
cap = cv2.VideoCapture(0) #passes video input into local variable for use in detection.  Replace with '1' if you have two cameras.

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[5]:

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[10]:

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})





      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)


      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



     #parse detected objects to find matches with sample set of models we can print

      threshold = 0.7
      for index, value in enumerate(classes[0]):

        if scores[0, index] > threshold:
            objectDetected = (category_index.get(value)).get('name').encode('utf8')
            objectDetected = str(objectDetected)
            objectDetected = objectDetected.replace("'", "")
            objectDetected = objectDetected.replace("b", "")
            print(objectDetected)

            objects = ['cell phone', 'scissors'] #some sample objects to print

    #texts a matching object name to owner.

            if objectDetected in objects:
                account_sid = #put your account id as a string here
                auth_token = #put your auth token as a string here
                client = Client(account_sid, auth_token)

                message = client.messages.create(
                     to= #number receiving text as a string followed by ","
                     from_= #twilio number as a string followed by ","
                     body= #whatever you want the text to say,  can personalize using local variables.


                          )

                print(message.sid)
                print("Detected a Known Object, message sent.")
                time.sleep(10)

    #Send gcode of detected object to printer

                def removeComment(string):
                    if (string.find(';')==-1):
                        return string
                    else:
                        return string[:string.index(';')]

                # Open serial port

                ser = serial.Serial()
                ser.port = 'COM3'
                ser.baudrate = 115200
                ser.open()
                out = ser.readline()
                print('Opening Serial Port')

                # Open g-code file
                #f = open('/media/UNTITLED/shoulder.g','r');
                root = "b:\\"
                gcode_name = objectDetected + ".gcode"
                file_name = os.path.join(root, "SpaghettiDetector", "models-master", "research", "object_detection", "gcode", gcode_name) #replace with file path to your gcode
                print(file_name)
                f = open(file_name)
                print('Opening gcode file')

                # Wake up

                time.sleep(2)   # Wait for printer to initialize
                ser.flushInput()  # Flush startup text in serial input
                print('Sending gcode')

                # Stream g-code, parses each line in file
                for line in f:
                    l = removeComment(line)
                    l = l.strip() # Strip all EOL characters for streaming
                    if  (l.isspace()==False and len(l)>0) :
                        print('Sending: ' + l)
                        block = "b'"+ l + '\n'
                        ser.write(block.encode()) # Send g-code block
                        grbl_out = ser.readline() # Wait for response with carriage return


                # Wait here until printing is finished to close serial port and file.
                #raw_input("  Press <Enter> to exit.")

                # Close file and serial port
                f.close()
                ser.close()



