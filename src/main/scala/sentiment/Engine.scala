package sentiment

import io.prediction.controller.{Engine, EngineFactory}

case class Query(sentence: String) extends Serializable

case class Result(sentiment: String, confidence: Double) extends Serializable

object SentimentEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("algo" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
