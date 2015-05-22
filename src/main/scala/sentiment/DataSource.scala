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
      ("we have to stop global warming!", PredictedResult(sentiment="Yes", confidence = 1)),
      ("global warming is real", PredictedResult(sentiment="Yes", confidence = 1)),
      ("global warming does not exist", PredictedResult(sentiment="No", confidence = 1)),
      ("i think there is not global warming", PredictedResult(sentiment="No", confidence = 1)),
      ("global warming is fake", PredictedResult(sentiment="No", confidence = 1)),
      ("Jet fuel can't melt steel beams", PredictedResult(sentiment="NA", confidence = 1))
    ))
  }
}

case class TrainingData(
  val sentences: Array[(String, PredictedResult)]
) extends Serializable
