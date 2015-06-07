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
  batchSize: Int
) extends Params

class DataSource(params: DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, Result] with RegexParsers {

  @transient lazy val logger = Logger[this.type]

  override def readTraining(sc: SparkContext): TrainingData = {
    val data = readPTB("data/train.txt")
    val batches = data.take((data.size * params.fraction).toInt).grouped(params.batchSize)
    val rdd = sc.parallelize(batches.toSeq, 64)
    TrainingData(rdd)
  }

  override def readEval(sc: SparkContext): Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, Result)])] = {
    val data = readPTB("data/train.txt")
    val batches = data.take((data.size * params.fraction).toInt).grouped(params.batchSize)
    val training = sc.parallelize(batches.toSeq)
    val eval = sc.parallelize(readPTB("data/test.txt")).map(t => (Query(Right(t._1)), t._2))
    Seq((TrainingData(training), new EmptyEvaluationInfo(), eval))
  }

  def readPTB(path: String): Seq[(Tree[Unit, String], Result)] = {
    val data = Source
      .fromFile(path)
      .getLines()
      .toSeq

    data
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
      .map(t => (Tree.accums.map(t)(_ => ()), Result(t)))
  }

  def sentVec(label: String): Sentiment.Vector[Double] =
    Sentiment.space.tabulatePartial(Map(Sentiment.classes(label) -> 1d))

  def tree: Parser[Tree[Sentiment.Vector[Double], String]] =
    ("(" ~ string ~ string ~ ")" ^^ {
      case _ ~ label ~ word ~ _ => Leaf(sentVec(label), word)
    }) | ("(" ~ string ~ tree ~ tree ~ ")" ^^ {
      case _ ~ label ~ left ~ right ~ _ => Node(left, sentVec(label), right)
    })

  def string: Parser[String] = "[^\\s()]+"r
}

case class TrainingData(
  get: RDD[Seq[(Tree[Unit, String], Result)]]
) extends Serializable
