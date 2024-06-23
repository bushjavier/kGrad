## kGrad

kGrad is a lightweight Kotlin library designed for educational purposes, inspired by Andrej Karpathy's Micrograd. It aims to facilitate understanding of automatic differentiation and backpropagation, essential components of deep learning. kGrad provides a clear and concise implementation of these core concepts, avoiding the complexity of larger frameworks like TensorFlow or PyTorch.

Key Features of kGrad:

- Automatic Differentiation: kGrad performs automatic differentiation, a method to automatically compute the gradients of functions. This is crucial for training neural networks as it allows for the efficient calculation of gradients needed for optimization algorithms like gradient descent.

- Backpropagation: The library includes Jupyter notebooks with examples of backpropagation, the algorithm used to compute the gradient of the loss function with respect to each weight in the neural network. This is fundamental for updating the weights to minimize the loss during training.

- Scalar-Based Computation: kGrad operates on scalar values, making it easier to understand the flow of gradients and how operations are chained together. This simplicity is beneficial for educational purposes, allowing users to grasp the basics of computational graphs and gradient flow.


Example Usage:

```
// Define four scalar values
val a = Value(2.0)
val b = Value(-3.0)
val c = Value(10.0)
val f = Value(-2.0)

// Perform some operations
val d = a * b
val e = d + c
val l = e * f

// Compute the gradient
l.backward()

// Result
println(l) // Output: -8.0

// Access the gradients
println(a.grad)  // Output: 6.0
println(b.grad)  // Output: -4.0
println(c.grad)  // Output: -2.0
println(f.grad)  // Output: 4.0

l.plot().render(Format.PNG).toFile(java.io.File("output/graph-example.png"))

```
The image generated "graph-example.png" illustrates the computational graph created for the expression described above.

![alt text](https://github.com/bushjavier/kGrad/blob/main/graph-example.png?raw=true)


