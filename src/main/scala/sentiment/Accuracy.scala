package sentiment

import io.prediction.controller._

case class Accuracy ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double =
    if (predicted.sentiment == actual.sentiment) 1.0 else 0.0
}

object SentimentEvaluation extends Evaluation with EngineParamsGenerator {
  engineEvaluator = (
    SentimentEngine(),
    MetricEvaluator(
      metric = Accuracy()
    ))

  engineParamsList = Seq(EngineParams(algorithmParamsList = Seq(("algo", EmptyParams()))))
}
