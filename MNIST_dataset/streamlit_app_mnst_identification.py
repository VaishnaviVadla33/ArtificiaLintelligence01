import streamlit as st
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

# Load MNIST dataset to preprocess and train if needed
from keras.datasets import mnist
from keras.utils import to_categorical

# Title and instructions
st.title("Handwritten Digit Recognition")
st.write("Use your webcam to show a handwritten digit, and the app will identify it.")

# Sidebar options
option = st.sidebar.selectbox("Options", ["Live Webcam Detection", "Train Model"])

# Function to preprocess MNIST dataset
def preprocess_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test

# Train and save model if option selected
if option == "Train Model":
    st.write("Training Model on MNIST Dataset...")
    X_train, y_train, X_test, y_test = preprocess_mnist_data()
    
    # Define a simple NN
    nn = Sequential([
        Dense(512, activation="relu", input_shape=(784,)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    nn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Train the model
    nn.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))
    
    # Save the trained model
    nn.save("mnist_model.h5")
    st.write("Model trained and saved as `mnist_model.h5`!")

# Webcam-based digit recognition
if option == "Live Webcam Detection":
    # Load the trained model
    try:
        model = load_model("mnist_model.h5")
        st.write("Loaded pre-trained model successfully.")
    except:
        st.error("Pre-trained model not found. Please train the model first.")
        st.stop()

    # Start webcam
    st.write("Starting webcam...")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    # Process webcam frames
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access webcam.")
            break

        # Convert to grayscale and resize to 28x28
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize and reshape input for the model
        processed = resized.reshape(1, 784) / 255.0

        # Predict digit
        prediction = model.predict(processed)
        predicted_digit = np.argmax(prediction)

        # Display predictions on the video feed
        cv2.putText(frame, f"Prediction: {predicted_digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    st.write("Stopped webcam.")
