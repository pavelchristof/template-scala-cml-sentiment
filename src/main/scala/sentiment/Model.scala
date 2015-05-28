package sentiment

import edu.stanford.nlp.trees.Tree
import io.prediction.controller.P2LAlgorithm
import io.prediction.data.storage.BiMap
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger

import scala.util.Random
import scalaz.Functor
import scalaz.Scalaz._

import cml._
import cml.algebra.traits._
import cml.algebra.Instances._
import cml.algebra.Constant
import cml.models._

case class RNTN[WordVec[_]] (
  implicit concrete: Concrete[WordVec]
) extends Model[({type T[A] = Constant[Tree, A]})#T, WordVec] {
  val wordVecPair = algebra.Product[WordVec, WordVec]()(concrete, concrete)
  val wordVecMap = Function[String, WordVec]()(Enumerate.string(Enumerate.char), concrete)
  val combiner = Chain3[wordVecPair.Type, algebra.Product[wordVecPair.Type, wordVecPair.Type]#Type, WordVec, WordVec](
    Duplicate[wordVecPair.Type] : Model[wordVecPair.Type, algebra.Product[wordVecPair.Type, wordVecPair.Type]#Type],
    LinAffinMap[wordVecPair.Type, wordVecPair.Type, WordVec]()(wordVecPair, wordVecPair, concrete)
      : Model[algebra.Product[wordVecPair.Type, wordVecPair.Type]#Type, WordVec],
    Pointwise[WordVec](AnalyticMap.tanh)(concrete)
  )

  type Type[A] = (wordVecMap.Type[A], combiner.Type[A])

  override def apply[A](inst: Type[A])(input: Constant[Tree, A])(implicit field: Analytic[A]): WordVec[A] = {
    val node = input.value
    if (node.isLeaf) {
      wordVecMap(inst._1)(Constant(node.value()))
    } else {
      node.children
        .map(t => apply(inst)(Constant(t)))
        .reduceLeft[WordVec[A]](combiner(inst._2)(_, _))
    }
  }

  override implicit val space =
    algebra.Product.locallyConcrete[wordVecMap.Type, combiner.Type](wordVecMap.space, combiner.space)

  override def fill[A](x: => A)(implicit a: Additive[A]): Type[A] =
    (Map().withDefault(_ => concrete.point(a.zero)), combiner.fill(x))
}

case class WrappedTree[A] (get: Tree) extends Traversable[String] {
  override def foreach[U](f: (String) => U): Unit = {
    f(get.value())
    for (ch <- get.children()) {
      WrappedTree(ch).foreach(f)
    }
  }

  override def reduce[A1 >: String](op: (A1, A1) => A1): A1 = {
    if (get.isLeaf) {
      get.value()
    } else {
      get.children().map(ch => WrappedTree(ch).reduce(op)).reduce(op)
    }
  }
}

object Model {
  import cml.models._
  import cml.optimization._
  import shapeless.Nat

  /*
   * First we declare types and implicits that we'll use in the model.
   */
  implicit val treeFunctor = new Functor[WrappedTree] {
    override def map[A, B](fa: WrappedTree[A])(f: (A) => B): WrappedTree[B] = WrappedTree(fa.get)
  }
  implicit val wordVec = algebra.Vector(Nat(5))
  implicit val wordVecPair = algebra.Product[wordVec.Type, wordVec.Type]
  implicit val sentimentVec = algebra.Vector(Nat(3))

  /**
   * A recursive neural tensor network model.
   */
  val model: Model[WrappedTree, sentimentVec.Type] = Chain3(
    MapReduce[String, WrappedTree, wordVec.Type](
      map = Function[String, wordVec.Type]()(Enumerate.string(Enumerate.char), implicitly),
      reduce = Chain3[wordVecPair.Type, algebra.Product[wordVecPair.Type, wordVecPair.Type]#Type, wordVec.Type, wordVec.Type](
        Duplicate[wordVecPair.Type] : Model[wordVecPair.Type, algebra.Product[wordVecPair.Type, wordVecPair.Type]#Type],
        LinAffinMap[wordVecPair.Type, wordVecPair.Type, wordVec.Type]
          : Model[algebra.Product[wordVecPair.Type, wordVecPair.Type]#Type, wordVec.Type],
        Pointwise[wordVec.Type](AnalyticMap.tanh)
      )
    ),

    // First we reduce the sentence tree into a vector.
 //   RNTN[wordVec.Type]: Model[WrappedTree, wordVec.Type],
    // The next layer maps the word vector to a sentiment vector.
    AffineMap[wordVec.Type, sentimentVec.Type],
    // Finally we apply softmax.
    Softmax[sentimentVec.Type]
  )

  /**
   * The cost function for our model.
   */
  val costFun = new CostFun[WrappedTree, sentimentVec.Type] {
    /**
     * This function scores a single sample (input, expected output and actual output triple).
     *
     * The cost for the whole data set is assumed to be the mean of scores for each sample.
     */
    override def scoreSample[A](sample: Sample[WrappedTree[A], sentimentVec.Type[A]])(implicit an: Analytic[A]): A = {
      import an.analyticSyntax._
      val eps = fromDouble(0.0001)

      // Softmax regression cost function. We add epsilon to prevent taking the log of 0.
      val j = ^(sample.expected, sample.actual){ case (e, a) =>
        e * (a + eps).log
      }
      -sentimentVec.sum(j)
    }

    /**
     * Computes the regularization term for a model instance.
     */
    override def regularization[V[_], A](instance: V[A])(implicit an: Analytic[A], space: LocallyConcrete[V]): A =
      an.mul(an.fromDouble(7), space.quadrance(instance))
  }

  /**
   * Now we create an optimizer that will train our model. The MultiOpt optimizer is a higher-order optimizer that
   * launches multiple optimizers in parallel and collects their results. We use gradient descent as our base optimizer.
   *
   * Gradient descent takes an optional gradient transformer, which is a function applied to the gradient before a
   * step is made. Here we apply numerical stabilization and then AdaGrad to automatically take care of step size.
   */
  val optimizer = GradientDescent(
      model,
      iterations = 50,
      gradTrans = Stabilize.andThen(AdaGrad).andThen(Scale(0.1))
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
      val in: WrappedTree[Double] = WrappedTree(tree)
      val index = sentiments(expected)
      val out: sentimentVec.Type[Double] = sentimentVec.tabulateLC(Map(index -> 1d))
      (in, out)
    }}

    // Value that the new model instances will be filled with.
    val rng = new Random()
    val filler = () => rng.nextDouble * 2d - 1d

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

    val noIndex = sentiments("No")
    val yesIndex = sentiments("Yes")

    def process(node: Tree): SentenceTree = {
      val sent = model(modelInstance.get)(WrappedTree(node))
      SentenceTree(
        label = node.value(),
        yes = sentimentVec.index(sent)(yesIndex),
        no = sentimentVec.index(sent)(noIndex),
        children = node.children().map(process)
      )
    }

    val tree = Parser(query.sentence)
    process(tree)
  }

  /**
   * A mapping between sentiment vector indices and sentiment labels.
   */
  val sentiments = BiMap.stringInt(Array("No", "Yes", "NA"))
}
