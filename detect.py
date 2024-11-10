import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

while True:
    frame = picam2.capture_array()

    # Preprocess the image
    input_shape = input_details[0]['shape']
    print(input_shape)
    input_data = cv2.resize(frame, (input_shape[3], input_shape[2]))
    input_data = np.transpose(input_data, [2, 0, 1])
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float32) / 255.0
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
	
    # Draw bounding boxes on the frame
    for detection in output_data:
        ymin, xmin, ymax, xmax, class_id, score, _, _ = detection
        if score > 0.5:  # Confidence threshold
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            cv2.putText(frame, f'Class: {int(class_id)}, Score: {score:.2f}', 
                        (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
picam2.stop()
cv2.destroyAllWindows()
