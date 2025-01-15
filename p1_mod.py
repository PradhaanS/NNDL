import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

w1 = np.random.rand()
w2 = np.random.rand()
bias = np.random.rand()
learning_rate = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(output):
    return output * (1 - output)

for epoch in range(5000):
    for i in range(4):
        z = x[i][0] * w1 + x[i][1] * w2 + bias
        result = sigmoid(z)
        error = y[i] - result
        d_result = error * sigmoid_derivative(result)
        dw1 = d_result * x[i][0]
        dw2 = d_result * x[i][1]
        dbias = d_result
        w1 += learning_rate * dw1
        w2 += learning_rate * dw2
        bias += learning_rate * dbias

print("Final weights:", w1, w2)
print("Final bias:", bias)

for i in range(4):
    z = x[i][0] * w1 + x[i][1] * w2 + bias
    result = sigmoid(z)
    print(f"Input: {x[i]}, Output: {result:.4f}, Predicted: {1 if result >= 0.5 else 0}")
