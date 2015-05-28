package sentiment

import edu.stanford.nlp.trees.Tree
import io.prediction.controller.P2LAlgorithm
import io.prediction.data.storage.BiMap
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger

import scala.util.Random
import scalaz.Scalaz._

import cml._
import cml.algebra.traits._
import cml.algebra.Instances._
import cml.algebra.Constant
import cml.models._

case class RNN[WordVec[_]] (
  implicit concrete: Concrete[WordVec]
) extends Model[({type T[A] = Constant[Tree, A]})#T, WordVec] {
  val wordVecPair = algebra.Product[WordVec, WordVec]()(concrete, concrete)
  val wordVecMap = Function[String, WordVec]()(Enumerate.string(Enumerate.char), concrete)
  val reducer = Chain2(
    AffineMap[wordVecPair.Type, WordVec]()(wordVecPair, concrete),
    Pointwise[WordVec](AnalyticMap.tanh)(concrete)
  )

  type Type[A] = (wordVecMap.Type[A], reducer.Type[A])

  override def apply[A](inst: Type[A])(input: Constant[Tree, A])(implicit field: Analytic[A]): WordVec[A] = {
    val node = input.value
    if (node.isLeaf) {
      wordVecMap(inst._1)(Constant(node.value()))
    } else {
      node.children
        .map(t => apply(inst)(Constant(t)))
        .reduceLeft[WordVec[A]](reducer(inst._2)(_, _))
    }
  }

  override implicit val space: LocallyConcrete[Type] =
    algebra.Product.locallyConcrete[wordVecMap.Type, reducer.Type](wordVecMap.space, reducer.space)

  override def fill[A](x: => A)(implicit a: Additive[A]): Type[A] =
    (wordVecMap.fill(x), reducer.fill(x))
}

object Model {
  import cml.models._
  import cml.optimization._
  import shapeless.Nat

  /*
   * First we declare types and implicits that we'll use in the model.
   */
  type WrappedTree[A] = Constant[Tree, A]
  implicit val treeFunctor = Constant.functor[Tree]
  implicit val wordVector = algebra.Vector(Nat(5))
  implicit val output = algebra.Scalar

  /**
   * A recursive neural network model.
   */
  val model: Model[WrappedTree, output.Type] = Chain3(
    // First we reduce the sentence tree into a vector.
    RNN[wordVector.Type]: Model[WrappedTree, wordVector.Type],
    // The next layer maps the word vector to a sentiment vector.
    AffineMap[wordVector.Type, output.Type],
    // Finally we apply a modified sigmoid that assumes values in the range [-1, 1].
    Pointwise[output.Type](new AnalyticMap {
      override def apply[F](x: F)(implicit f: Analytic[F]): F = {
        import f.analyticSyntax._
        _1 - _2 / (_1 + (-x).exp)
      }
    })
  )

  /**
   * The cost function for our model.
   */
  val costFun = new CostFun[WrappedTree, output.Type] {
    /**
     * This function scores a single sample (input, expected output and actual output triple).
     *
     * The cost for the whole data set is assumed to be the mean of scores for each sample.
     */
    override def scoreSample[A](sample: Sample[WrappedTree[A], output.Type[A]])(implicit an: Analytic[A]): A = {
      import an.analyticSyntax._
      (sample.expected - sample.actual).square
    }

    /**
     * Computes the regularization term for a model instance.
     */
    override def regularization[V[_], A](instance: V[A])(implicit an: Analytic[A], space: LocallyConcrete[V]): A =
      an.mul(an.fromDouble(0.001), space.quadrance(instance))
  }

  /**
   * Now we create an optimizer that will train our model. The MultiOpt optimizer is a higher-order optimizer that
   * launches multiple optimizers in parallel and collects their results. We use gradient descent as our base optimizer.
   *
   * Gradient descent takes an optional gradient transformer, which is a function applied to the gradient before a
   * step is made. Here we apply numerical stabilization and then AdaGrad to automatically take care of step size.
   */
  val optimizer = MultiOpt(
    populationSize = 4,
    optimizer = GradientDescent(
      model,
      iterations = 70,
      gradTrans = Stabilize.andThen(AdaGrad).andThen(Scale(0.1))
    )
  )

  /**
   * We need to declare what automatic differentiation engine should be used. Backpropagation is the best.
   */
  implicit val diffEngine = ad.Backward
}

/**
 * Wraps a model instance.
 */
case class ModelInstance (
  get: Model.model.Type[Double]
)

class Algorithm extends P2LAlgorithm[PreparedData, ModelInstance, Query, SentenceTree] {

  @transient lazy val logger = Logger[this.type]

  /**
   * Trains a model instance.
   */
  def train(sc: SparkContext, data: PreparedData): ModelInstance = {
    import Model._

    println(data.sentences.length)

    // First we have to convert the data set to our model's input format.
    val dataSet = data.sentences.map { case (tree, expected) => {
      val in: WrappedTree[Double] = Constant(tree)
      (in, expected)
    }}

    // Value that the new model instances will be filled with.
    val rng = new Random()
    val filler = () => rng.nextDouble * 0.2d - 0.1d

    // Run the optimizer!
    val inst =
      optimizer[Double](
        // This is the starting population, in case we want to improve existing instances.
        // We do not have any trained model instances so we just pass an empty vector.
        population = Vector(),
        data = dataSet,
        costFun = costFun,
        default = filler())
      // Optimizer returns a vector of (cost, instance) pairs. Here we select the instance with the lowest cost.
      .minBy(_._1)._2
      // And unfortunately we have to explicitly cast it to the right type. This is because scala doesn't know
      // that model.Type = optimizer.model.Type, even thought it is quite obvious to us since model == optimizer.model.
      .asInstanceOf[model.Type[Double]]

    ModelInstance(inst)
  }

  /**
   * Queries the model.
   */
  def predict(modelInstance: ModelInstance, query: Query): SentenceTree = {
    import Model._

    def process(node: Tree): SentenceTree =
      SentenceTree(
        label = node.value(),
        sentiment = model(modelInstance.get)(Constant(node)),
        children = node.children().map(process)
      )

    val tree = Parser(query.sentence)
    process(tree)
  }
}
