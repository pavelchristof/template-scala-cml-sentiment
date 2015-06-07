package sentiment

import cml._
import cml.algebra.Floating._
import cml.algebra._
import cml.models._
import cml.optimization._
import grizzled.slf4j.Logger
import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext

import scala.util.Random
import scalaz.Semigroup

case class RNTNParams (
  wordVecSize: Int,
  stepSize: Double,
  iterations: Int,
  regularizationCoeff: Double,
  noise: Double
) extends Params

class RNTN (params: RNTNParams) extends Serializable {
  // First declare the size of our vectors. We use RuntimeNat here because the size depend on algorithm parameters.
  val wordVecSize = algebra.RuntimeNat(params.wordVecSize)

  // Now lets declare the types of vectors that we'll be using.
  type WordVec[A] = Vec[wordVecSize.Type, A]
  type SentimentVec[A] = Sentiment.Vector[A]
  type WordVecPair[A] = (WordVec[A], WordVec[A])
  type WordVecQuad[A] = (WordVecPair[A], WordVecPair[A])
  type Word[A] = String
  type InputTree[A] = Tree[Unit, String]
  type WordVecTree[A] = Tree[WordVec[A], String]
  type OutputTree[A] = Tree[SentimentVec[A], String]

  // We have to find the required implicits by hand because Scala doesn't support type classes.
  implicit val wordVecSpace = Vec.cartesian(wordVecSize())
  implicit val sentimentVecSpace = Sentiment.space
  implicit val wordVecPairSpace = Cartesian.product[WordVec, WordVec]
  implicit val wordVecQuadSpace = Cartesian.product[WordVecPair, WordVecPair]
  implicit val inputTreeFunctor: ZeroFunctor[InputTree] = ZeroFunctor.const
  implicit val outputTreeFunctor: ZeroFunctor[OutputTree] =
    ZeroFunctor.compose[({type T[A] = Tree[A, String]})#T, SentimentVec](Tree.accumsZero[String], sentimentVecSpace)

  /**
   * A recursive neural tensor network model.
   */
  val model = Chain2[InputTree, WordVecTree, OutputTree](
    // In the first part of the algorithm we map each word to a vector and then propagate
    // the vectors up the tree using a merge function.
    AccumulateTree[Word, WordVec](
      // The function that maps words to vectors.
      inject = SetMap[String, WordVec],
      // Merge function, taking a pair of vectors and returning a single vector.
      reduce = Chain3[WordVecPair, WordVecQuad, WordVec, WordVec](
        // Duplicate takes a single argument x (of type WordVecPair) and returns (x, x).
        Duplicate[WordVecPair],
        // LinAffinMap is a function on two arguments: linear in the first and affine in the second. This is
        // equivalent to the sum of a bilinear form (on both arguments) and a linear form on the first argument.
        // The type parameters are argument types and the result type.
        LinAffinMap[WordVecPair, WordVecPair, WordVec],
        // Apply the activaton function pointwise over the word vector.
        Pointwise[WordVec](AnalyticMap.tanh)
      )
    ) : Model[InputTree, WordVecTree],

    // In the second part we map over the tree to classify the word vectors.
    BifunctorMap[Tree, WordVec, SentimentVec, Word, Word](
      // Word vectors go thought a softmax classifier.
      left = Chain2(
        AffineMap[WordVec, SentimentVec],
        Softmax[SentimentVec]
      ),
      // Words are unchanged.
      right = Identity[Word]
    )
  )

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
      def j(e: SentimentVec[A], a: SentimentVec[A]): A =
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

  /**
   * We need to declare what automatic differentiation engine should be used. Backpropagation is the best.
   */
  implicit val diffEngine = ad.Backward
}

class Algorithm (
  params: RNTNParams
) extends P2LAlgorithm[TrainingData, Any, Query, Result] {
  @transient lazy val logger = Logger[this.type]

  /**
   * Trains a model instance.
   */
  override def train(sc: SparkContext, data: TrainingData): Any = {
    val rntn = new RNTN(params)
    import rntn._

    val dataSet = data.get.map(_.map(x => (x._1, x._2.sentence)))

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
    val rntn = new RNTN(params)
    import rntn._

    val input = query.sentence match {
      case Left(s) => Parser(s)
      case Right(t) => t
    }
    val inst = instUntyped.asInstanceOf[model.Type[Double]]

    Result(model(inst)(input))
  }
}
