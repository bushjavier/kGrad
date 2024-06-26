# kGrad

kGrad is a lightweight Kotlin library designed for educational purposes, inspired by Andrej Karpathy's Micrograd. It aims to facilitate understanding of automatic differentiation and backpropagation, essential components of deep learning. kGrad provides a clear and concise implementation of these core concepts, avoiding the complexity of larger frameworks like TensorFlow or PyTorch.

### Key Features of kGrad:

- Automatic Differentiation: kGrad performs automatic differentiation, a method to automatically compute the gradients of functions. This is crucial for training neural networks as it allows for the efficient calculation of gradients needed for optimization algorithms like gradient descent.

- Backpropagation: The library includes Jupyter notebooks with examples of backpropagation, the algorithm used to compute the gradient of the loss function with respect to each weight in the neural network. This is fundamental for updating the weights to minimize the loss during training.

- Scalar-Based Computation: kGrad operates on scalar values, making it easier to understand the flow of gradients and how operations are chained together. This simplicity is beneficial for educational purposes, allowing users to grasp the basics of computational graphs and gradient flow.

### Examples:

#### Automatic differentiation

This example demonstrates using kGrad to construct arithmetic expressions and calculate the corresponding gradient with respect to values.

```kotlin
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

l.plot().render(Format.PNG).toFile(java.io.File("graph-example.png"))

```
The image generated "graph-example.png" illustrates the computational graph created for the expression described above.

![alt text](https://github.com/bushjavier/kGrad/blob/main/graph-example.png?raw=true)

#### Training a Neural Network

This example demonstrates the training process of a Multi-Layer Perceptron (MLP), illustrating the steps involved in training a simple neural network.

```kotlin
// Define a Multi-Layer Perceptron (MLP) with 3 input nodes, 2 hidden layers with 4 nodes each, and 1 output node.
val n = MLP(3, listOf(4, 4, 1))

// Input dataset
val xs = listOf(
    listOf(Value(2.0), Value(3.0), Value(-1.0)),
    listOf(Value(3.0), Value(-1.0), Value(0.5)),
    listOf(Value(0.5), Value(1.0), Value(1.0)),
    listOf(Value(1.0), Value(1.0), Value(-1.0)),
)

// Desired output targets for the dataset
val ys = listOf(Value(1.0), Value(-1.0), Value(-1.0), Value(1.0))

// Learning rate for gradient descent optimization
val learningRate = 0.1

// Training loop
for (i in 0..100) {

    // Forward pass: Apply the network on each input to get predictions
    val yPred = xs.flatMap { n(it) }

    // Compute the loss between predicted and actual outputs
    val loss = ys.zip(yPred) { y, pred -> (y - pred).pow(2.0) }.reduce { a, b -> a + b }

    // Zero out gradients in preparation for backpropagation
    n.zeroGrad()

    // Perform backpropagation to compute gradients
    loss.backward()

    // Update parameters using gradient descent
    n.parameters().forEach {
        it.data += -learningRate * it.grad
    }

    // Print current training progress
    println("$i\tloss: ${loss.data}")
}
```

### A Jupyter notebook for constructing kGrad from scratch.

[This](https://github.com/bushjavier/kGrad/blob/main/Writing%20kGrad%20from%20scratch%20notebook.ipynb) Jupyter notebook guides the creation of kGrad, through these key steps:

- **Theoretical Foundation**: It begins with a review of calculus theory, focusing on partial derivatives.

- **Value Class**: A class called Value is defined to handle scalar values and their corresponding gradients.

- **Operations Implementation**: Basic arithmetic operations (like addition and multiplication) and more complex mathematical functions are implemented, ensuring that gradients are computed correctly.

- **Backward Propagation**: Techniques such as backward propagation are developed to compute gradients using the chain rule and other derivative rules efficiently.

- **Implementation of Neural Network Components**: Classes like Neuron, Layer, and MLP (Multilayer Perceptron) are implemented to build and train neural networks.

- **Integration**: Finally, all components are integrated to create a comprehensive automatic differentiation framework, which forms the basis of kGrad.

https://github.com/bushjavier/kGrad/blob/main/Writing%20kGrad%20from%20scratch%20notebook.ipynb

The notebook is in Kotlin please use [this guide](https://github.com/Kotlin/kotlin-jupyter#installation) to run the notebook locally 
