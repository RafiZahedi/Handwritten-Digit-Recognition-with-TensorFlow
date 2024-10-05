import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter import filedialog

print("Loading Model")
# Load the trained model
model = tf.keras.models.load_model('digits.model.h5')
print("Model Loaded")

def process_image(image_path):
    """ Preprocess the image for prediction """
    # Read the image using OpenCV in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not open or find the image: {image_path}")

    # Resize the image to 28x28 pixels (MNIST image size)
    img = cv2.resize(img, (28, 28))

    # Invert colors (if needed for proper prediction, as MNIST is white digits on black background)
    img = np.invert(img)

    # Normalize the image (to match the training data normalization)
    img = tf.keras.utils.normalize(img, axis=1)

    # Reshape the image to match the input format of the model: (1, 28, 28)
    img = img.reshape(1, 28, 28)

    return img


def predict_digit(model, img):
    """ Predict the digit from the processed image """
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    return predicted_digit


def open_file_dialog():
    """ Open a file dialog to select an image """
    # Create a Tkinter root window (it will be hidden)
    root = Tk()
    root.withdraw() # is it necc? idk man

    # Open the file dialog and allow user to select an image file
    file_path = filedialog.askopenfilename(
        title="Select Handwritten Digit Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )

    # Close the root window after file selection
    root.destroy()

    return file_path


def main():
    while True:
        # Open file dialog to select the image
        image_path = open_file_dialog()

        # If no file is selected exit the loop
        if not image_path:
            print("No file selected. Exiting the program...")
            break

        try:
            # Process the image
            img = process_image(image_path)

            # Predict the digit
            predicted_digit = predict_digit(model, img)

            # Output the predicted digit
            print(f"The number is probably a {predicted_digit}")

            # Display the image for confirmation
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f"Predicted: {predicted_digit}")
            plt.show()

        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with a valid image file.")


if __name__ == "__main__":
    main()
