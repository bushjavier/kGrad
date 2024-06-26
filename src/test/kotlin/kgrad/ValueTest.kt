package kgrad

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals


internal class ValueTest {

    @Test
    fun testOperations() {

        val x1 = Value(-4.0)
        val z1 = x1 * 2.0 + 2.0 + x1
        val q1 = z1.tanh() + z1 * x1
        val h1 = (z1 * z1).tanh()
        val y1 = h1 + q1 + q1 * x1
        y1.backward()

        //Compare with Deep Java Library using Pytorch engine
        NDManager.newBaseManager().use { manager ->
            val x: NDArray = manager.create(-4.0)
            x.setRequiresGradient(true)

            Engine.getInstance().newGradientCollector().use { gc ->
                val z = x.mul(2.0).add(2.0).add(x)
                val q = z.tanh().add(z.mul(x))
                val h = z.mul(z).tanh()
                val y = h.add(q).add(q.mul(x))
                gc.backward(y)

                //forward pass went well
                assertEquals(y.getDouble(), y1.data)

                //backward pass went well
                assertEquals(x.gradient.getDouble(), x1.grad)
            }
        }
    }

    @Test
    fun testMoreOperations() {

        val a1 = Value(-4.0)
        val b1 = Value(2.0)
        var c1 = a1 + b1
        var d1 = a1 * b1 + b1.pow(3.0)
        c1 = c1 + c1 + 1.0
        c1 = c1 + c1 + 1.0 + (-a1)
        d1 = d1 + d1 * 2.0 + (b1 + a1).tanh()
        d1 = d1 + d1 * 3.0 + (b1 - a1).tanh()
        val e1 = c1 - d1
        val f1 = e1.pow(2.0)
        var g1 = f1 / 2.0
        g1 +=  f1 / 10.0
        g1.backward()

        //Compare with Deep Java Library using Pytorch engine
        NDManager.newBaseManager().use { manager ->
            val a: NDArray = manager.create(-4.0)
            a.setRequiresGradient(true)

            val b: NDArray = manager.create(2.0)
            b.setRequiresGradient(true)

            Engine.getInstance().newGradientCollector().use { gc ->
                var c = a.add(b)
                var d = a.mul(b).add(b.pow(3.0))
                c = c.add(c.add(1.0))
                c = c.add(c.add(1.0)).add(a.neg())
                d = d.add(d.mul(2.0).add(b.add(a).tanh()))
                d = d.add(d.mul(3.0).add(b.sub(a).tanh()))
                val e = c.sub(d)
                val f = e.pow(2.0)
                var g = f.div(2.0)
                g = g.add(f.div(10.0))
                gc.backward(g)

                val tolerance = 1e-9

                //forward pass went well
                assert(abs(g.getDouble() - g1.data) < tolerance)

                //backward pass went well
                assert(abs(a.gradient.getDouble() - a1.grad) < tolerance)
                assert(abs(b.gradient.getDouble() - b1.grad) < tolerance)
            }
        }
    }
}