package kgrad

import guru.nidi.graphviz.attribute.Label
import guru.nidi.graphviz.attribute.Rank
import guru.nidi.graphviz.attribute.Records
import guru.nidi.graphviz.engine.Format
import guru.nidi.graphviz.engine.Graphviz
import guru.nidi.graphviz.engine.GraphvizJdkEngine
import guru.nidi.graphviz.model.Factory
import guru.nidi.graphviz.model.MutableNode
import java.awt.image.BufferedImage
import kotlin.math.pow


class Value(var data: Double, var label: String? = null) {

    private var children: List<Value> = listOf()
    private var usedOperator: String? = null
    var grad = 0.0
    private var backwardFunction: () -> Unit = {}

    operator fun plus(other: Value): Value = Value(this.data + other.data).also { plusResult ->
        plusResult.usedOperator = "+"
        plusResult.children = listOf(this, other)

        plusResult.backwardFunction = fun() {
            this.grad += 1.0 * plusResult.grad
            other.grad += 1.0 * plusResult.grad
        }
    }

    operator fun times(other: Value): Value = Value(this.data * other.data).also { timesResult ->
        timesResult.usedOperator = "*"
        timesResult.children = listOf(this, other)

        timesResult.backwardFunction = fun() {
            this.grad += other.data * timesResult.grad
            other.grad += this.data * timesResult.grad
        }
    }

    operator fun minus(other: Value): Value = Value(this.data - other.data).also { minusResult ->
        minusResult.usedOperator = "-"
        minusResult.children = listOf(this, other)

        minusResult.backwardFunction = fun() {
            this.grad += 1.0 * minusResult.grad
            other.grad += -1.0 * minusResult.grad
        }
    }

    operator fun plus(other: Double): Value = plus(Value(other))

    operator fun times(other: Double): Value = times(Value(other))

    operator fun minus(other: Double): Value = minus(Value(other))

    operator fun unaryMinus(): Value = this * -1.0

    operator fun div(other: Value): Value = this * other.pow(-1.0)

    operator fun div(other: Double): Value = this * other.pow(-1.0)

    fun pow(other: Double): Value = Value(this.data.pow(other)).also { powResult ->
        powResult.usedOperator = "^"
        powResult.children = listOf(this)

        powResult.backwardFunction = fun() {
            this.grad += other * this.data.pow(other - 1) * powResult.grad
        }
    }

    fun exp(): Value = Value(kotlin.math.exp(data)).also { expResult ->
        expResult.usedOperator = "exp"
        expResult.children = listOf(this)

        expResult.backwardFunction = fun() {
            this.grad += expResult.data * expResult.grad
        }
    }

    fun tanh(): Value {
        val x = this.data
        val tanh = ((kotlin.math.exp(2 * x) - 1)) / ((kotlin.math.exp(2 * x) + 1))
        return Value(tanh).also { tanhResult ->
            tanhResult.usedOperator = "tanh"
            tanhResult.children = listOf(this)
            tanhResult.backwardFunction = fun() {
                this.grad += (1 - tanh.pow(2)) * tanhResult.grad
            }
        }
    }

    fun leakyRelu(alpha: Double = 0.01): Value {
        val leakyReluData = if (data > 0) data else alpha * data
        return Value(leakyReluData).also { leakyReluResult ->
            leakyReluResult.usedOperator = "leakyRelu"
            leakyReluResult.children = listOf(this)
            leakyReluResult.backwardFunction = fun() {
                val gradInput = if (data > 0) 1.0 else alpha
                this.grad += gradInput * leakyReluResult.grad
            }
        }
    }

    fun relu(): Value = Value(if (this.data < 0.0) 0.0 else this.data).also { reluResult ->
        reluResult.usedOperator = "ReLU"
        reluResult.children = listOf(this)

        reluResult.backwardFunction = fun() {
            this.grad += (if (this.data > 0.0) 1.0 * reluResult.grad else 0.0)
        }
    }

    fun backward() {
        val topo = mutableListOf<Value>()
        val visited = mutableSetOf<Value>()
        fun buildTopo(v: Value) {
            if (!visited.contains(v)) {
                visited.add(v)
                for (child in v.children) {
                    buildTopo(child)
                }
                topo.add(v)
            }
        }

        buildTopo(this)

        this.grad = 1.0
        for (v in topo.reversed()) {
            v.backwardFunction()
        }
    }

    override fun toString(): String {
        return "Value(data=$data)"
    }

    fun plot(): BufferedImage {

        Graphviz.useEngine(GraphvizJdkEngine())

        val nodes = mutableSetOf<Value>()
        val edges = mutableSetOf<Pair<Value, Value>>()

        fun traverseGraph(v: Value) {
            if (!nodes.contains(v)) {
                nodes.add(v)
                v.children.forEach {
                    edges.add(Pair(it, v))
                    traverseGraph(it)
                }
            }
        }
        traverseGraph(this)

        val graph = Factory.mutGraph().setDirected(true).graphAttrs().add(Rank.dir(Rank.RankDir.LEFT_TO_RIGHT))

        nodes.forEach { node ->
            val valueNode = Factory.mutNode(System.identityHashCode(node).toString()).add(
                Records.of(
                    Records.turn(
                        *listOfNotNull(
                            node.label?.let { Records.rec(it) },
                            Records.rec("data:  " + node.data.toString()),
                            Records.rec("grad:  " + node.grad),
                        ).toTypedArray()
                    )
                )
            )
            graph.add(valueNode)
            if (node.usedOperator != null) {
                val operatorNode = Factory.mutNode(System.identityHashCode(node).toString() + node.usedOperator)
                node.usedOperator?.let { operatorNode.add(Label.of(it)) }
                operatorNode.addLink(valueNode)
                graph.add(operatorNode)
            }
        }
        edges.forEach {
            var graphNodeFirst: MutableNode? = null
            var graphNodeSecondOp: MutableNode? = null
            for (n in graph.nodes()) {
                if (n.name().toString() == System.identityHashCode(it.first).toString()) {
                    graphNodeFirst = n
                }
            }
            for (n in graph.nodes()) {
                if (n.name().toString() == System.identityHashCode(it.second).toString() + it.second.usedOperator) {
                    graphNodeSecondOp = n
                }
            }
            graphNodeSecondOp?.let { so ->
                graphNodeFirst?.addLink(so)
            }
        }
        return Graphviz.fromGraph(graph).render(Format.PNG).toImage()
    }
}