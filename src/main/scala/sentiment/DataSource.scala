package sentiment

import cml._
import cml.algebra.Floating._
import grizzled.slf4j.Logger
import io.prediction.controller._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source
import scala.util.parsing.combinator.RegexParsers

case class DataSourceParams(
  fraction: Double,
  batchSize: Int = 10
) extends Params

class DataSource(params: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, Result] with RegexParsers {

  @transient lazy val logger = Logger[this.type]

  override def readTraining(sc: SparkContext): TrainingData = {
    TrainingData(readPTB(sc, "data/train.txt").sample(withReplacement = false, fraction = params.fraction).cache())
  }

  override def readEval(sc: SparkContext): Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, Result)])] = {
    val training = readPTB(sc, "data/train.txt").sample(withReplacement = false, fraction = params.fraction).cache()
    val eval = readPTB(sc, "data/test.txt").map(t => (t._1: Query, t._2))
    Seq((TrainingData(training), new EmptyEvaluationInfo(), eval))
  }

  def readPTB(sc: SparkContext, path: String): RDD[(TreeQuery, Result)] = {
    val data = Source
      .fromFile(path)
      .getLines()
      .toSeq

    sc.parallelize(data, data.size / params.batchSize)
      .map(parse(tree, _))
      .flatMap {
        case Success(v, _) => Some(v)
        case NoSuccess(msg, _) => {
          println(msg)
          None
        }
      }
      // Filter out neutral sentences like the RNTN paper does.
      .filter(t => Sentiment.choose(t.accum) != "2")
      .map(t => (TreeQuery(Tree.accums.map(t)(_ => ())), Result(t)))
  }

  def sentVec(label: String): Sentiment.Vector[Double] =
    Sentiment.space.tabulatePartial(Map(Sentiment.classes(label) -> 1d))

  def tree: Parser[Tree[Sentiment.Vector[Double], String]] =
    ("(" ~ string ~ string ~ ")" ^^ {
      case _ ~ label ~ word ~ _ => Leaf(sentVec(label), word)
    }) | ("(" ~ string ~ tree ~ tree ~ ")" ^^ {
      case _ ~ label ~ left ~ right ~ _ => Node(left, sentVec(label), right)
    })

  def string: Parser[String] = "[^\\s()]+"r //"[!-'*-/0-9:-@A-Z\\[-`a-z{-~\u0128-\u0255]+"r
}

case class TrainingData(
  get: RDD[(TreeQuery, Result)]
) extends Serializable
