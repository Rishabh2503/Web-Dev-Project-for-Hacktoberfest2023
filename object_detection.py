import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model (You need to have a model file and labels)
model_path = 'path_to_your_model.pb'  # Replace with the path to your model file
label_path = 'path_to_your_labels.txt'  # Replace with the path to your label file

with open(label_path, 'r') as f:
    labels = f.read().strip().split('\n')

model = tf.saved_model.load(model_path)

# Load an image for object detection
image_path = 'path_to_your_image.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to the input size expected by the model
input_size = 300  # Adjust to match your model's input size
image = cv2.resize(image, (input_size, input_size))

# Make predictions on the image
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]
detections = model(input_tensor)

# Process the detections
boxes = detections['detection_boxes'][0].numpy()
scores = detections['detection_scores'][0].numpy()
classes = detections['detection_classes'][0].numpy().astype(np.int32)

for i in range(len(boxes)):
    if scores[i] > 0.5:  # You can adjust the confidence threshold as needed
        class_name = labels[classes[i]]
        ymin, xmin, ymax, xmax = boxes[i]
        image_height, image_width, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        left = int(xmin * image_width)
        top = int(ymin * image_height)
        right = int(xmax * image_width)
        bottom = int(ymax * image_height)

        # Draw bounding boxes and labels on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, class_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes and labels
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
