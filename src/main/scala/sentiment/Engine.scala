package sentiment

import io.prediction.controller.{Engine, EngineFactory}

case class Query(
  sentence: String
) extends Serializable

case class SentenceTree (
  label: String,

  yes: Double,
  no: Double,

  children: Seq[SentenceTree]
) extends Serializable

object SentimentEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("rntn" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
