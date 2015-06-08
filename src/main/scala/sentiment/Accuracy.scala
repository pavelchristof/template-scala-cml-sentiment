package sentiment

import cml.Tree
import io.prediction.controller._

object AccuracyUtils {
  def equal(u: Sentiment.Vector[Double], v: Sentiment.Vector[Double]): Double =
    if (Sentiment.choose(u) == Sentiment.choose(v)) 1d else 0d

  def equalBin(u: Sentiment.Vector[Double], v: Sentiment.Vector[Double]): Double = {
    val a = Sentiment.choose(u)
    val b = Sentiment.choose(v)
    if (toBin(a) == toBin(b)) 1d else 0d
  }

  def toBin(s: String): Int =
    s match {
      case "0" => -1
      case "1" => -1
      case "2" => 0
      case "3" => 1
      case "4" => 1
    }
}

case class AccuracyRoot ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double =
    AccuracyUtils.equal(predicted.sentence.accum, actual.sentence.accum)
}

case class AccuracyAll ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double = {
    val zipped = predicted.sentence.zip(actual.sentence)
    val scored = Tree.accums.map(zipped)(p => AccuracyUtils.equal(p._1, p._2))
    val list = Tree.accums.toList(scored)
    list.sum / list.size
  }
}

case class AccuracyRootBinary ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double =
    AccuracyUtils.equalBin(predicted.sentence.accum, actual.sentence.accum)
}

case class AccuracyAllBinary ()
  extends AverageMetric[EmptyEvaluationInfo, Query, Result, Result] {
  def calculate(query: Query, predicted: Result, actual: Result): Double = {
    val zipped = predicted.sentence.zip(actual.sentence)
    val scored = Tree.accums.map(zipped)(p => AccuracyUtils.equalBin(p._1, p._2))
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
        AccuracyRoot(),
        AccuracyAllBinary(),
        AccuracyRootBinary()
      )
    ))

  engineParamsList = Seq(
    EngineParams(
      dataSourceParams = DataSourceParams(fraction = 1, batchSize = 25),
      algorithmParamsList = Seq(
        ("mm", MMParams(
          vecSize = 15,
          stepSize = 0.03,
          regularizationCoeff = 1e-3,
          iterations = 20,
          noise = 0.1 // Better then 1.0
        ))
      )
    ),
    EngineParams(
      dataSourceParams = DataSourceParams(fraction = 1, batchSize = 25),
      algorithmParamsList = Seq(
        ("rntn", RNTNParams(
          wordVecSize = 7,
          stepSize = 0.1,
          regularizationCoeff = 1e-3,
          iterations = 50,
          noise = 0.1 // Better then 1.0
        ))
      )
    ),
    EngineParams(
      dataSourceParams = DataSourceParams(fraction = 1, batchSize = 25),
      algorithmParamsList = Seq(
        ("rnn", RNNParams(
          wordVecSize = 7,
          stepSize = 0.1,
          regularizationCoeff = 1e-3,
          iterations = 50,
          noise = 0.1 // Better then 1.0
        ))
      )
    )
  )
}
