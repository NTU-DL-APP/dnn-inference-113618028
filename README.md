[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lkpPbEwb)
# DNN Inference Test

## Introduction 

In this test, you will classify the **Fashion-MNIST** dataset using only **NumPy** for inference. You may train your model using any platform, such as Google Colab or your local machine. Once training is complete, save your model into `.h5` format and then convert into the two formats:
- **Architecture fie**: `fashion_mnist.json`  
- **Weights file**: `fashion_mnist.npz`

A basic neural network forward-pass pipeline is provided in [`nn_predict.py`](nn_predict.py). You are required to implement the missing `relu()` and `softmax()` functions. The Fashion-MNIST dataset is available in the [`./data`](./data) folder.


## Implementation Workflow

1. Implement the `relu()` and `softmax()` functions in [`nn_predict.py`](nn_predict.py).
2. Design and train your own neural network using TensorFlow.  
   > **Note:** The provided `nn_predict.py` supports only **ReLU**, **Softmax**, and **fully connected (Dense)** layers. You are welcome to extend the code to support additional layer types.
3. Train your model to achieve good accuracy.
4. Save your trained model in `.h5` format, then extract:
   - Weights to `fashion_mnist.npz`
   - Architecture to `fashion_mnist.json`  
   You can refer to this [demo notebook](https://colab.research.google.com/drive/1zHH3-EujP9P1bF39xiEpHdbc8lIC-xdj?usp=sharing) for guidance.
5. Upload the `fashion_mnist.json` and `fashion_mnist.npz` files to the [`./model`](./model) folder.
6. Commit your changes to trigger the test.
7. Iterate until your model achieves the desired performance.


## Test Cases

Your score will be based on the functions you fulfilled and the accuracy of your model. Here are the credit sheet:

| Task | Points |
|----------|----------|
| Implement `relu()`     | 10     |
| Implement `softmax()` | 10     |
| Upload valid model and run inference | 50     |
| Test Accuracy > 0.8 | 10     |
| Test Accuracy > 0.82 | 2     |
| Test Accuracy > 0.84 | 2     |
| Test Accuracy > 0.86 | 2     |
| Test Accuracy > 0.88 | 3     |
| Test Accuracy > 0.9 | 3     |
| Test Accuracy > 0.91 | 4     |
| Test Accuracy > 0.92 | 4     |

## Note

Modify the test file `modet_test.py` will fail the test and consider as cheat.
