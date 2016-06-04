package neural

import jdk.internal.util.xml.impl.Input
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.mutable.{ListBuffer, Seq}
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

  def evaluate(testData: List[(INDArray, INDArray)]) = ???

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
    val nabla = biasWeightPairs.map(zeroPair)
    var activation = batch._1
    val activations = ListBuffer(batch._1)
    val zs = new ListBuffer[INDArray]
    biasWeightPairs.foreach{pair =>
      val z = pair._2.dot(activation) + pair._1
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    }

    var delta = costDerivative(activations.reverse, batch._2) ( sigmoidPrime(zs.reverse))

  }

  def costDerivative(reverse: ListBuffer[INDArray], _2: INDArray) = ???

  def sigmoidPrime(reverse: ListBuffer[INDArray]) = {
    reverse.map(el => sigmoid(el)*(sigmoid(el)*(-1)+1))
  }


}
