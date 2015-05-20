package sentiment

import grizzled.slf4j.Logger
import io.prediction.controller.{EmptyActualResult, EmptyEvaluationInfo, PDataSource, Params}
import org.apache.spark.SparkContext

case class DataSourceParams(appName: String) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    TrainingData(Array(
      ("I think global warming is a real issue", PredictedResult(sentiment="Yes", confidence = 1)),
      ("Jet fuel can't melt steel beams", PredictedResult(sentiment="No", confidence = 1))
    ))
  }
}

case class TrainingData(
  val sentences: Array[(String, PredictedResult)]
) extends Serializable
