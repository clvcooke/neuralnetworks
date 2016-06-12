package neural

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.util.Random


/**
  * Created by Colin on 6/1/2016.
  */
class Network(sizes: List[Int]) {


  val numLayers = sizes.length
  var biasWeightPairs = getRandomBiases.zip(getRandomWeights)

  def getRandomBiases = sizes.tail.map(number => Nd4j.randn(number, 1))

  def getRandomWeights = sizes.reverse.tail.zip(sizes.tail).map(pair => Nd4j.randn(pair._2, pair._1))

  def feedForward(input: INDArray): INDArray = {
    biasWeightPairs.foldLeft(input)((a, pair) => sigmoid(pair._2.mmul(a).add(pair._1)))
  }


  def zeroPair(pair: (INDArray, INDArray)) = {
    //splatting necessary to make java var args work with scala arrays
    (Nd4j.zeros(pair._1.shape(): _*), Nd4j.zeros(pair._2.shape(): _*))
  }

  def evaluate(testData: List[(INDArray, INDArray)]) = {
    testData.count(data => Nd4j.argMax(feedForward(data._1)) == data._2)
  }

  //Stochastic Gradient Descent
  def SGD(trainingData: List[(INDArray, INDArray)], epochs: Int, miniBatchSize: Int, eta: Double, testData: List[(INDArray, INDArray)] = null): Unit = {
    for (i <- epochs) {
      new Random().shuffle(trainingData).sliding(miniBatchSize).foreach(updateMiniBatch(_, eta))
      print(s"Epoch $i" + (if (testData != null) s": ${evaluate(testData)} / ${testData.length}" else " complete"))
    }
  }

  def addInnerPairs(pair1: (INDArray, INDArray), pair2: (INDArray, INDArray)) = {
    (pair1._1.add(pair2._1), pair1._2.add(pair2._2))
  }

  def mulInnerPairs(pair1: (INDArray, INDArray), pair2: (INDArray, INDArray)) = {
    (pair1._1.mul(pair2._1), pair1._2.mul(pair2._2))
  }

  def adjustInnerPairs(pair1: (INDArray, INDArray), pair2: (INDArray, INDArray), eta: Double, miniBatchLength: Int) = {
    (pair1._1 - (pair2._1 * (eta / miniBatchLength)), pair1._2 - (pair2._2 * (eta / miniBatchLength)))
  }

  //  Update the network's weights and biases by applying
  //        gradient descent using backpropagation to a single mini batch.
  //        The "mini_batch" is a list of tuples "(x, y)", and "eta"
  //        is the learning rate.
  def updateMiniBatch(miniBatch: List[(INDArray, INDArray)], eta: Double)(implicit k: String): Unit = {
    val nabla = miniBatch.foldLeft(biasWeightPairs.map(zeroPair))((n, pair) => n.zip(backProp(pair)).map(p => addInnerPairs(p._1, p._2)))
    biasWeightPairs = biasWeightPairs.zip(nabla).map(p => adjustInnerPairs(p._1, p._2, eta, miniBatch.length))
  }


  /*
    self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
   */

  def backProp(batch: (INDArray, INDArray)): List[(INDArray, INDArray)] = {
    var activation = batch._1
    val zsa = biasWeightPairs.map { pair =>
      val z = pair._2.dot(activation) + pair._1
      activation = sigmoid(z)
      (z, activation)
    }.reverse
    var delta = costDerivative(zsa.head._2, batch._2) * sigmoidPrime(zsa.head._1)
    zsa.drop(1).zip(biasWeightPairs).map { p =>
      delta = p._2._2.dot(delta) * sigmoidPrime(p._1._1)
      (delta, delta.dot(p._1._2.transpose()))
    }
    /*
    x, y
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
      */


  }

  def costDerivative(el: INDArray, batchArr: INDArray): INDArray = el-batchArr

  def sigmoidPrime(el: INDArray): INDArray = {
    sigmoid(el) * (sigmoid(el) * (-1) + 1)
  }


}