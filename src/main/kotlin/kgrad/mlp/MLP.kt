package kgrad.mlp

import kgrad.Value

class MLP(inputs: Int, outputs: List<Int>) {
    private val layers: List<Layer> = outputs.indices.map { i ->
        val layerSizes = listOf(inputs).plus(outputs)
        Layer(layerSizes[i], layerSizes[i + 1])
    }

    operator fun invoke(x: List<Value>): List<Value> {

        var tmpValues = x

        layers.forEach {
            tmpValues = it(tmpValues)
        }
        return tmpValues
    }

    fun parameters(): List<Value> {
        return layers.flatMap { it.parameters() }
    }

    fun zeroGrad() {
        parameters().forEach {
            it.grad = 0.0
        }
    }
}