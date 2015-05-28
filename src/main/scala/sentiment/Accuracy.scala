package sentiment

import io.prediction.controller._

case class Accuracy ()
  extends AverageMetric[EmptyEvaluationInfo, Query, SentenceTree, Double] {
  def calculate(query: Query, predicted: SentenceTree, actual: Double): Double =
    if (toLabel(predicted.sentiment) == toLabel(actual)) 1.0 else 0.0

  def toLabel(sent: Double) =
    if (sent * 3 >= 1)
      "Yes"
    else if (sent * 3 <= -1)
      "No"
    else
      "N/A"
}

object SentimentEvaluation extends Evaluation with EngineParamsGenerator {
  engineEvaluator = (
    SentimentEngine(),
    MetricEvaluator(
      metric = Accuracy()
    ))

  engineParamsList = Seq(EngineParams(algorithmParamsList = Seq(("algo", EmptyParams()))))
}
