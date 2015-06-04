package sentiment

import cml.Tree
import io.prediction.controller._

object AccuracyUtils {
  def equalInd(u: Sentiment.Vector[Double], v: Sentiment.Vector[Double]): Double =
    if (Sentiment.choose(u) == Sentiment.choose(v))
      1d
    else
      0d
}

case class AccuracyRoot ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double =
    AccuracyUtils.equalInd(predicted.sentence.accum, actual.sentence.accum)
}

case class AccuracyAll ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double = {
    val zipped = predicted.sentence.zip(actual.sentence)
    val scored = Tree.accums.map(zipped)(p => AccuracyUtils.equalInd(p._1, p._2))
    val list = Tree.accums.toList(scored)
    list.sum / list.size
  }
}

object SentimentEvaluation extends Evaluation with EngineParamsGenerator {
  engineEvaluator = (
    SentimentEngine(),
    MetricEvaluator(
      metric = AccuracyAll(),
      otherMetrics = Seq(
        AccuracyRoot()
      )
    ))

  engineParamsList = for (reg <- Seq(1d, 1e-6))
    yield EngineParams(
      dataSourceParams = DataSourceParams(fraction = 1),
      algorithmParamsList = Seq(("rntn", RNTNParams(
        wordVecSize = 5,
        stepSize = 0.03,
        regularizationCoeff = reg,
        iterations = 1000,
        noise = 0.1 // Better then 1.0
      )))
    )
}
