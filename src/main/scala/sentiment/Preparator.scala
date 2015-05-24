package sentiment

import io.prediction.controller.PPreparator
import io.prediction.data.storage.Event
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scalaz.Tree

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(sentences = trainingData.sentences.map { case (a, b) => (Parser(a.sentence), b) })
  }
}

class PreparedData(
  val sentences: Array[(Tree[String], Result)]
) extends Serializable