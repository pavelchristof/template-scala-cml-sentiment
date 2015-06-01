package sentiment

import edu.stanford.nlp.trees.{SimpleTree, Tree}
import io.prediction.controller.{LAlgorithm, Params, P2LAlgorithm}
import io.prediction.data.storage.BiMap
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger

import scala.util.Random
import scalaz._

import cml._
import cml.algebra._
import cml.models._
import cml.optimization._
import Floating._

case class RNTNParams (
  wordVecSize: Int,
  stepSize: Double,
  iterations: Int,
  regularizationCoeff: Double,
  noise: Double
) extends Params

class RNTN (params: RNTNParams) {
  /**
   * A mapping between sentiment vector indices and sentiment labels.
   */
  val sentiments = BiMap.stringInt(Array("No", "Yes", "NA"))

  // First declare the sizes of our vectors. We use RuntimeNat here because the sizes depend on algorithm parameters.
  val wordVecSize = algebra.RuntimeNat(params.wordVecSize)
  val sentimentVecSize = algebra.RuntimeNat(sentiments.size)

  // Now lets declare the types of vectors that we'll be using.
  type WordVec[A] = Vec[wordVecSize.Type, A]
  type SentimentVec[A] = Vec[sentimentVecSize.Type, A]
  type WordVecPair[A] = (WordVec[A], WordVec[A])
  type WordVecQuad[A] = (WordVecPair[A], WordVecPair[A])

  // We have to find the required implicits by hand because Scala doesn't support type classes.
  implicit val wordVecSpace = Cartesian.vec(wordVecSize())
  implicit val sentimentVecSpace = Cartesian.vec(sentimentVecSize())
  implicit val wordVecPairSpace = Cartesian.product[WordVec, WordVec]
  implicit val wordVecQuadSpace = Cartesian.product[WordVecPair, WordVecPair]

  // Sentence tree not take a type parameter (because it doesn't depend on the field type). However model's
  // input has to be a functor. We'll wrap the tree in a constant functor to get around that.
  type WrappedTree[A] = Const[Tree, A]
  implicit val treeFunctor: ZeroFunctor[WrappedTree] = ZeroFunctor.const(new SimpleTree())

  // Trees can be folded (reduced). We have to use the monomorphic variant, because Tree is not a functor - node values
  // are strings and cannot be mapped to any other type.
  implicit val treeMonoFoldable = new MonoFoldable1[Tree, String] {
    override def foldMap1[S](v: Tree)(inj: (String) => S, op: (S, S) => S): S =
      if (v.isLeaf) {
        inj(v.value())
      } else {
        v.children().map(foldMap1(_)(inj, op)).reduceLeft(op)
      }
  }

  /**
   * A recursive neural tensor network model.
   */
  val model = Chain3(
    // Map-reduce the tree getting a word vector as the result.
    // The first parameter is the container type, the second is element type and the last is result type.
    MonoMapReduce[Tree, String, WordVec](
      // We map each word to a word vector using a hash map.
      map = HashMap[String, WordVec],
      // Then we reduce the pair with a model that takes two word vectors and returns one.
      // Type inference fails here and we have to provide the types of all immediate values.
      reduce = Chain3[WordVecPair, WordVecQuad, WordVec, WordVec](
        // Duplicate takes a single argument x (of type wordVecType.Type) and returns (x, x).
        Duplicate[WordVecPair],
        // LinAffinMap is a function on two arguments: linear in the first and affine in the second. This is
        // equivalent to the sum of a bilinear form (on both arguments) and a linear form on the first argument.
        // The type parameters are argument types and the result type.
        LinAffinMap[WordVecPair, WordVecPair, WordVec],
        // Apply the activaton function pointwise over the word vector.
        Pointwise[WordVec](AnalyticMap.tanh)
      )
    ) : Model[WrappedTree, WordVec],
    // We have a word vector now - we still have to classify it.
    // The next layer maps the word vector to a sentiment vector using an affine map (i.e. linear map with bias).
    AffineMap[WordVec, SentimentVec],
    // Finally we apply softmax.
    Softmax[SentimentVec]
  )

  /**
   * The cost function for our model.
   */
  val costFun = new CostFun[WrappedTree, SentimentVec] {
    /**
     * This function scores a single sample (input, expected output and actual output triple).
     *
     * The cost for the whole data set is assumed to be the mean of scores for each sample.
     */
    override def scoreSample[A](sample: Sample[WrappedTree[A], SentimentVec[A]])(implicit an: Analytic[A]): A = {
      import an.analyticSyntax._
      val eps = fromDouble(1e-9)

      // Softmax regression cost function. We add epsilon to prevent taking the log of 0.
      val j = sentimentVecSpace.apply2(sample.expected, sample.actual)((e, a) => e * (a + eps).log)
      -sentimentVecSpace.sum(j)
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
  val optimizer = GradientDescent(
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
) extends LAlgorithm[PreparedData, Any, Query, SentenceTree] {
  @transient lazy val logger = Logger[this.type]

  /**
   * Trains a model instance.
   */
  override def train(data: PreparedData): Any = {
    val rntn = new RNTN(params)
    import rntn._

    // First we have to convert the data set to our model's input format.
    val dataSet = data.sentences.map { case (tree, expected) => {
      val in: WrappedTree[Double] = Const(tree)
      val index = sentiments(expected)
      val out = sentimentVecSpace.tabulatePartial(Map(index -> 1d))
      (in, out)
    }}

    // Value that the new model instances will be filled with.
    val rng = new Random()
    val noise = () => (rng.nextDouble * 2d - 1d) * params.noise

    // Find the finite subspace of the model that we'll be using.
    val subspace = optimizer.model.restrict(dataSet, costFun)
    println(s"Model dimension: ${subspace.space.dim}")

    // Run the optimizer!
    optimizer[Double](
      // This is the starting population, in case we want to improve existing instances.
      // We do not have any trained model instances so we just pass an empty vector.
      population = Vector(),
      data = dataSet,
      costFun = costFun,
      noise = noise(),
      subspace = subspace)
      // Optimizer returns a vector of (cost, instance) pairs. Here we select the instance with the lowest cost.
      .minBy(_._1)._2
  }

  /**
   * Queries the model.
   */
  override def predict(inst: Any, query: Query): SentenceTree = {
    val rntn = new RNTN(params)
    import rntn._

    val noIndex = sentiments("No")
    val yesIndex = sentiments("Yes")

    def process(node: Tree): SentenceTree = {
      val sent = model(inst.asInstanceOf[model.Type[Double]])(Const(node))
      SentenceTree(
        label = node.value(),
        yes = sentimentVecSpace.index(sent)(yesIndex),
        no = sentimentVecSpace.index(sent)(noIndex),
        children = node.children().map(process)
      )
    }

    val tree = Parser(query.sentence)
    process(tree)
  }
}
