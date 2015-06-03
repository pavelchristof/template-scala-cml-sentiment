package sentiment

import java.io.File

import grizzled.slf4j.Logger
import io.prediction.controller._
import org.apache.spark.SparkContext
import com.github.tototoshi.csv._
import org.apache.spark.rdd.RDD

import scala.util.Random

case class DataSourceParams(
  fraction: Double
) extends Params

class DataSource(params: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, String] {

  @transient lazy val logger = Logger[this.type]

  override def readTraining(sc: SparkContext): TrainingData = {
    TrainingData(readData(sc))
  }

  override def readEval(sc: SparkContext): Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, String)])] = {
    val Array(training, eval) = readData(sc).randomSplit(Array(0.6, 0.4))
    Seq((TrainingData(training.cache()), new EmptyEvaluationInfo(), eval.cache()))
  }

  def readData(sc: SparkContext): RDD[(Query, String)] = {
    val reader = CSVReader.open(new File("data/tweets.csv"))
    val data = for (row <- reader.all())
      yield (Query(row(0)), row(1))
    sc.parallelize(data, data.size / 100).sample(withReplacement = false, fraction = params.fraction)
  }
}

case class TrainingData(
  sentences: RDD[(Query, String)]
) extends Serializable
