package sentiment

import java.io.File

import grizzled.slf4j.Logger
import io.prediction.controller._
import org.apache.spark.SparkContext
import com.github.tototoshi.csv._
import org.apache.spark.rdd.RDD

import scala.util.Random

class DataSource
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, Double] {

  @transient lazy val logger = Logger[this.type]

  override def readTraining(sc: SparkContext): TrainingData = {
    TrainingData(readData().take(300))
  }

  override def readEval(sc: SparkContext): Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, Double)])] = {
    val data = readData()

    val (training, eval) = data.splitAt(4500)
    val rdd = sc.parallelize(eval)
    rdd.cache()

    Seq((TrainingData(training), new EmptyEvaluationInfo(), rdd))
  }

  def readData(): Array[(Query, Double)] = {
    val reader = CSVReader.open(new File("data/tweets.csv"))
    val data = for (row <- reader.all())
      yield (Query(row(0)), getSentiment(row(1), row(2).toDouble))
    val rng = new Random()
    rng.shuffle(data).toArray
  }

  def getSentiment(label: String, confidence: Double): Double = label match {
    case "No" => -1
    case "NA" => 0
    case "Yes" => 1
  }
}

case class TrainingData(
  sentences: Array[(Query, Double)]
) extends Serializable
