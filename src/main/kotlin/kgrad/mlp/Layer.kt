package kgrad.mlp

import kgrad.Value

class Layer(inputsNumber: Int, outputsNumber: Int) {

    private val neurons: List<Neuron> = List(outputsNumber) {
        Neuron(inputsNumber)
    }

    operator fun invoke(x: List<Value>): List<Value> {
        return neurons.map { it(x) }
    }

    fun parameters(): List<Value> {
        return neurons.flatMap { it.parameters() }
    }
}