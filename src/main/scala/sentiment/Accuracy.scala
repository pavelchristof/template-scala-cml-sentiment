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

  engineParamsList = for (
      reg <- Seq(0d, 0.01d);
      ns <- Seq(1d, 0.1d)
    ) yield EngineParams(
      dataSourceParams = DataSourceParams(fraction = 1.0),
      algorithmParamsList = Seq(("rntn", RNTNParams(
        wordVecSize = 5,
        stepSize = 0.01,
        regularizationCoeff = reg,
        iterations = 600,
        noise = ns
      )))
    )
}
