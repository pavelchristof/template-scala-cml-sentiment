package sentiment

import java.io.File

import grizzled.slf4j.Logger
import io.prediction.controller._
import com.github.tototoshi.csv._

import scala.util.Random

class DataSource
  extends LDataSource[TrainingData, EmptyEvaluationInfo, Query, String] {

  @transient lazy val logger = Logger[this.type]

  override def readTraining(): TrainingData = {
    TrainingData(readData().take(50))
  }

  override def readEval(): Seq[(TrainingData, EmptyEvaluationInfo, Seq[(Query, String)])] = {
    val data = readData().take(300)
    val (training, eval) = data.splitAt(50)
    Seq((TrainingData(training), new EmptyEvaluationInfo(), eval))
  }

  def readData(): Array[(Query, String)] = {
    val reader = CSVReader.open(new File("data/tweets.csv"))
    val data = for (row <- reader.all())
      yield (Query(row(0)), row(1))
    val rng = new Random()
    rng.shuffle(data).toArray
  }
}

case class TrainingData(
  sentences: Array[(Query, String)]
) extends Serializable
