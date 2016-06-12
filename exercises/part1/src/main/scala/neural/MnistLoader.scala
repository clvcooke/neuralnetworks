package neural

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.{ArrayBuffer, ListBuffer}


//Filename: mnist.pkl.gz


/**
  * Created by Colin on 6/6/2016.
  */
object MnistLoader {

  private def gzipInputStream(s: String) = new GZIPInputStream(new BufferedInputStream(new FileInputStream(s)))

  private def read32BitInt(i: GZIPInputStream) = i.read() * 8388608 /*2^23*/ + i.read() * 32768 /*2&15*/ + i.read() * 128 /*2^7*/ + i.read()


  def getMnistImageData(baseDirectory: String) = {


    val images = readImages("")
    val labels = readLabels("")

  }

  def readLabels(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfLabels = read32BitInt(g)
    val labels = new ListBuffer[Double]()
    for(_<- 1 to numberOfLabels){
      labels.append(g.read().toDouble)
    }
    labels.toArray
  }

  def readImages(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfImages = read32BitInt(g)
    val rows = read32BitInt(g)
    val cols = read32BitInt(g)
    val images = new ListBuffer[INDArray]()
    for (_ <- 1 to numberOfImages) {
      val arr = Nd4j.create(cols*rows)
      //data is organized row wise
      for (_ <- 1 to cols) {
        for (_ <- 1 to rows) {
          //read in the date into a cols*rows buffer
          arr.add(g.read())
        }
      }
      images.append(arr)
    }
    images.toArray
  }

}

