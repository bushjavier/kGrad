package kgrad.mlp

import kgrad.Value
import kotlin.random.Random

class Neuron(inputsNumber: Int) {

    private val weights: List<Value> = List(inputsNumber) { Value(Random.nextDouble(-1.0, 1.0), "MLP-Weight") }
    private val bias: Value = Value(Random.nextDouble(-1.0, 1.0), "MLP-Bias")

    operator fun invoke(x: List<Value>): Value {
        // out = tanh(w * x + b)
        val activation = weights.zip(x) { a, b -> a * b }.reduce { a, b -> a + b } + bias
        val out = activation.tanh()
        return out
    }

    fun parameters(): List<Value> {
        return weights.plus(bias)
    }
}