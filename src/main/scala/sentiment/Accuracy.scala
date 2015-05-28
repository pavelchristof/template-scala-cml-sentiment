package sentiment

import io.prediction.controller._

case class Accuracy ()
  extends AverageMetric[EmptyEvaluationInfo, Query, SentenceTree, String] {
  def calculate(query: Query, predicted: SentenceTree, actual: String): Double = {
    val p = Map(
      "Yes" -> predicted.yes,
      "No" -> predicted.no,
      "NA" -> (1 - predicted.yes - predicted.no)
    )

    if (p.maxBy(_._2)._1 == actual) 1d else 0d
  }
}

object SentimentEvaluation extends Evaluation with EngineParamsGenerator {
  engineEvaluator = (
    SentimentEngine(),
    MetricEvaluator(
      metric = Accuracy()
    ))

  engineParamsList = Seq(EngineParams(algorithmParamsList = Seq(("algo", EmptyParams()))))
}
