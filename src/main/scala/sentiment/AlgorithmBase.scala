package sentiment

import cml._
import cml.algebra._
import cml.optimization.{Scale, AdaGrad, Stabilize, StochasticGradientDescent}
import grizzled.slf4j.Logger
import io.prediction.controller.{Params, P2LAlgorithm}
import org.apache.spark.SparkContext
import cml.algebra.Floating._

import scala.util.Random
import scalaz.Semigroup

trait AlgorithmParams extends Params {
  val stepSize: Double
  val iterations: Int
  val regularizationCoeff: Double
  val noise: Double
}

abstract class AlgorithmBase (
  params: AlgorithmParams
) extends P2LAlgorithm[TrainingData, Any, Query, Result] {
  @transient lazy val logger = Logger[this.type]

  type Word[A] = String
  type InputTree[A] = Tree[Unit, String]
  type OutputTree[A] = Tree[Sentiment.Vector[A], String]

  implicit val inputTreeFunctor: Functor[InputTree] = Functor.const
  implicit val outputTreeFunctor: Functor[OutputTree] =
    Functor.compose[({type T[A] = Tree[A, String]})#T, Sentiment.Vector](Tree.accumsZero[String], Sentiment.space)
  implicit val sentimentVecSpace = Sentiment.space

  val model: Model[InputTree, OutputTree]

  /**
   * The cost function for our model.
   */
  val costFun = new CostFun[InputTree, OutputTree] {
    /**
     * This function scores a single sample (input, expected output and actual output triple).
     *
     * The cost for the whole data set is assumed to be the mean of scores for each sample.
     */
    override def scoreSample[A](sample: Sample[InputTree[A], OutputTree[A]])(implicit an: Analytic[A]): A = {
      import an.analyticSyntax._
      val eps = fromDouble(1e-9)

      // The cost function for single vectors.
      def j(e: Sentiment.Vector[A], a: Sentiment.Vector[A]): A =
        sentimentVecSpace.sum(sentimentVecSpace.apply2(e, a)((e, a) => e * (a + eps).log))

      // We sum errors for each node.
      val zipped = sample.expected.zip(sample.actual)
      - Tree.accums.foldMap1(zipped)(x => j(x._1, x._2))(new Semigroup[A] {
        override def append(f1: A, f2: => A): A = f1 + f2
      })
    }

    /**
     * Computes the regularization term for a model instance.
     */
    override def regularization[V[_], A](instance: V[A])(implicit an: Analytic[A], space: Normed[V]): A =
      an.mul(an.fromDouble(params.regularizationCoeff), space.quadrance(instance))
  }

  /**
   * We need to declare what automatic differentiation engine should be used. Backpropagation is the best.
   */
  implicit val diffEngine = ad.Backward

  /**
   * Trains a model instance.
   */
  override def train(sc: SparkContext, data: TrainingData): Any = {
    val dataSet = data.get.map(_.map(x => (x._1, x._2.sentence)))

    /**
     * An optimizer is used to train the model.
     *
     * Gradient descent takes an optional gradient transformer, which is a function applied to the gradient before a
     * step is made. Here we apply numerical stabilization and then AdaGrad, finally scaling the gradient.
     */
    val optimizer = StochasticGradientDescent(
      model,
      iterations = params.iterations,
      gradTrans = Stabilize.andThen(AdaGrad).andThen(Scale(params.stepSize))
    )

    // Value that the new model instances will be filled with.
    val rng = new Random() with Serializable
    val initialInst = optimizer.model.space.tabulate(_ =>
      (rng.nextDouble * 2d - 1d) * params.noise)

    // Run the optimizer!
    val t1 = System.currentTimeMillis()
    val inst = optimizer[Double](
      dataSet,
      costFun,
      initialInst)
    println(s"Optimization took ${System.currentTimeMillis() - t1}ms.")

    inst
  }

  /**
   * Queries the model.
   */
  override def predict(instUntyped: Any, query: Query): Result = {
    val input = query.sentence match {
      case Left(s) => Parser(s)
      case Right(t) => t
    }
    val inst = instUntyped.asInstanceOf[model.Params[Double]]

    Result(model(inst)(input))
  }
}
