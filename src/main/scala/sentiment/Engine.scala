package sentiment

import cml.Tree
import cml.algebra.{Cartesian, Vec, RuntimeNat}
import io.prediction.controller.{Engine, EngineFactory}
import io.prediction.data.storage.BiMap

case class Query (
  sentence: Either[String, Tree[Unit, String]]
) extends Serializable

case class Result (
  sentence: Tree[Sentiment.Vector[Double], String]
) extends Serializable

object Sentiment {
  /**
   * A mapping between sentiment vector indices and sentiment labels.
   */
  val classes = BiMap.stringInt(Array("0", "1", "2", "3", "4"))

  /**
   * The number of sentiment classes as a type.
   */
  val size = RuntimeNat(classes.size)

  /**
   * A vector of class probabilities.
   */
  type Vector[A] = Vec[size.Type, A]

  /**
   * The space of vectors of class probabilities.
   */
  implicit val space = Vec.cartesian(size())

  /**
   * Choose the label with the highest probability.
   */
  def choose[A](vec: Vector[A])(implicit ord: Ordering[A]): String =
    classes.inverse(vec.get.zipWithIndex.maxBy(_._1)._2)
}

object SentimentEngine extends EngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("rntn" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
