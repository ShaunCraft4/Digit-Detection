'''
Use the following code to get the trained model for MNIST in the form of .h5 file:


from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test = x_test.reshape(-1,28,28,1).astype("float32")/255.0

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save("digit_model.h5")
'''



import cv2
import numpy as np
from tensorflow import keras  #To load and use your trained digit recognition neural network.
from collections import deque #A double-ended queue used here as a buffer to smooth predictions over multiple frames.

#Load trained model
model = keras.models.load_model("digit_model.h5")

#Start webcam
cap = cv2.VideoCapture(0)

#Buffer to smooth predictions
prediction_buffer = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    #Adaptive thresholding for lighting robustness
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    #Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 1000:  #filter small noise
            digit = thresh[y:y+h, x:x+w]
            
            #Make digit square and center it
            size = max(w, h)
            square = np.zeros((size, size), dtype=np.uint8)
            if w > h:
                y_offset = (w - h) // 2
                square[y_offset:y_offset+h, 0:w] = digit
            else:
                x_offset = (h - w) // 2
                square[0:h, x_offset:x_offset+w] = digit
            
            #Resize to 28x28
            digit_resized = cv2.resize(square, (28,28))
            digit_resized = digit_resized.reshape(1,28,28,1).astype("float32")/255.0  #This is the final image created form the camera after all the editing
            
            #Predict
            prediction = model.predict(digit_resized, verbose=0) #The image is used by the model to predict its content by creating probabilities on what it could be
            prediction_class = np.argmax(prediction)  #The content with the highest probability is chosen as the prediction
            
            #Add to buffer and smooth prediction
            prediction_buffer.append(prediction_class)
            display_class = max(set(prediction_buffer), key=prediction_buffer.count)
            
            #Show prediction
            cv2.putText(frame, str(display_class), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

