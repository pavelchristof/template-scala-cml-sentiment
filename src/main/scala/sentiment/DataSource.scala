package sentiment

import java.io.File

import grizzled.slf4j.Logger
import io.prediction.controller._
import org.apache.spark.SparkContext
import com.github.tototoshi.csv._
import org.apache.spark.rdd.RDD

import scala.util.Random

class DataSource
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, Result] {

  @transient lazy val logger = Logger[this.type]

  override def readTraining(sc: SparkContext): TrainingData = {
    TrainingData(readData())
  }

  override def readEval(sc: SparkContext): Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, Result)])] = {
    val data = readData().take(100)

    val (training, eval) = data.splitAt((data.length * 0.6).toInt)
    val rdd = sc.parallelize(eval)
    rdd.cache()

    Seq((TrainingData(training), new EmptyEvaluationInfo(), rdd))
  }

  def readData(): Array[(Query, Result)] = {
    val reader = CSVReader.open(new File("data/tweets.csv"))
    val data = for (row <- reader.all())
      yield (Query(row(0)), Result(sentiment = row(1), confidence = row(2).toDouble))
    val rng = new Random()
    rng.shuffle(data).toArray
  }
}

case class TrainingData(
  sentences: Array[(Query, Result)]
) extends Serializable
