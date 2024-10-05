# Handwritten Digit Recognition with TensorFlow

## Description
This project implements a deep learning-based model to recognize handwritten digits from images. Using a trained neural network (based on the MNIST dataset), the program processes images of digits and predicts their numerical value. The project features image preprocessing with OpenCV and provides a simple graphical interface for image selection using Tkinter.

## Features
- **Deep Learning Model**: Utilizes a TensorFlow/Keras model trained on the MNIST dataset to predict digits.
- **Image Processing**: Converts user-selected images to grayscale, resizes, normalizes, and inverts them for accurate predictions.
- **GUI**: Tkinter-based file dialog for easy image selection.
- **Visualization**: Displays the processed image along with the predicted digit using Matplotlib.

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RafiZahedi/handwritten-digit-recognition.git
   ```
2. Navigate into the project directory:
   ```bash
   cd handwritten-digit-recognition
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the trained model (`digits.model.h5`) and place it in the project directory (link to model if hosted externally).

## Usage
1. Run the script to start the program:
   ```bash
   python src/predict.py
   ```
2. Select an image file of a handwritten digit.
3. The predicted digit will be displayed along with the processed image.


## Future Improvements
- Add support for batch processing of multiple images.
- Provide an option to train the model from scratch using the MNIST dataset.
- Improve the user interface for a better experience.

---

**Author:** [Rafi Zahedi](https://github.com/RafiZahedi)
