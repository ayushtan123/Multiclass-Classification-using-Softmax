# Multiclass-Classification-using-Softmax 💻

## Overview
This project utilizes a neural network for multiclass classification, specifically recognizing hand-written digits (0-9). The model is implemented using TensorFlow and includes the ReLU activation function and the Softmax function for improved accuracy.

## Features
- **Packages**: Numpy, Matplotlib, TensorFlow.
- **Activation Functions**: ReLU, Softmax.
- **Model Architecture**: Three-layer neural network with ReLU activation in hidden layers and linear activation in the output layer.

## Dataset
- **Size**: 5000 training examples.
- **Input**: 20x20 grayscale images unrolled into a 400-dimensional vector.
- **Labels**: 5000x1 vector indicating the digit (0-9) for each image.

## Model
- **Architecture**: Input layer (400 units), Hidden layers (25, 15 units with ReLU activation), Output layer (10 units with linear activation).
- **Training**: Softmax grouped with loss function, SparseCategoricalCrossentropy loss, Adam optimizer, 40 epochs.

## Usage
1. Import required packages.
2. Load dataset using `load_data()`.
3. Build the model using Keras Sequential model.
4. Compile the model with specified loss function and optimizer.
5. Train the model using `model.fit(X, y, epochs=40)`.
6. Make predictions using `model.predict(image)`.
7. Evaluate accuracy and visualize results.

## Conclusion
This project demonstrates the successful implementation of a neural network for digit recognition. The model achieves accurate predictions and provides a useful template for similar multiclass classification tasks.
