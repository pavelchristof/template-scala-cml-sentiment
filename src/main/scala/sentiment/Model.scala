package sentiment

import io.prediction.controller.P2LAlgorithm
import io.prediction.data.storage.BiMap
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger

import scala.util.Random
import scalaz._
import Scalaz._

import cml._
import cml.algebra.traits._
import cml.algebra.Instances._
import cml.algebra.Constant

object Model {
  import cml.models._
  import cml.optimization._
  import shapeless.Nat

  /*
   * First we declare types and implicits that we'll use in the model.
   */
  type Word[A] = Constant[String, A]
  implicit val wordConcrete = Constant.concrete[String]
  implicit val strings = Enumerate.string(Enumerate.char)
  implicit val wordVector = algebra.Vector(Nat(5))
  implicit val wordVectorPair = algebra.Product[wordVector.Type, wordVector.Type]
  implicit val wordTree = algebra.Compose[Tree, Word]
  implicit val wordTreeFunctor = wordTree.functor
  implicit val vectorTree = algebra.Compose[Tree, wordVector.Type]
  implicit val sentimentVec = algebra.Vector(Nat(3))

  /**
   * A recurisve neural network model.
   */
  val model: cml.Model[wordTree.Type, sentimentVec.Type] = Chain4(
    // Firstly map words to word vectors.
    FunctorMap[Tree, Word, wordVector.Type](
      Function[String, wordVector.Type]
    ) : cml.Model[wordTree.Type, vectorTree.Type],
    // Now fold the tree using matrix multiplication with bias (an affine map) and sigmoid activation.
    Reduce[Tree, wordVector.Type](Chain2(
      AffineMap[wordVectorPair.Type, wordVector.Type],
      Pointwise[wordVector.Type](AnalyticMap.tanh)
    )) : cml.Model[vectorTree.Type, wordVector.Type],
    // The next layer maps the word vector to a sentiment vector.
    AffineMap[wordVector.Type, sentimentVec.Type],
    // Finally we apply softmax to get probabilities.
    Softmax[sentimentVec.Type]
  )

  /**
   * The cost function for our model.
   */
  val costFun = new CostFun[wordTree.Type, sentimentVec.Type] {
    /**
     * This function scores a single sample (input, expected output and actual output triple).
     *
     * The cost for the whole data set is assumed to be the mean of scores for each sample.
     */
    override def scoreSample[A](sample: Sample[wordTree.Type[A], sentimentVec.Type[A]])(implicit an: Analytic[A]): A = {
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
      an.mul(an.fromDouble(0.01), space.quadrance(instance))
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
      iterations = 50,
      gradTrans = Stabilize.andThen(AdaGrad)
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

class Algorithm extends P2LAlgorithm[PreparedData, ModelInstance, Query, Result] {

  @transient lazy val logger = Logger[this.type]

  /**
   * Trains a model instance.
   */
  def train(sc: SparkContext, data: PreparedData): ModelInstance = {
    import Model._

    println(data.sentences.length)

    // First we have to convert the data set to our model's input format.
    val dataSet = data.sentences.map { case (tree, expected) => {
      val index = sentiments(expected.sentiment)
      val in: wordTree.Type[Double] = tree.map(Constant(_))
      val out: sentimentVec.Type[Double] = sentimentVec.tabulateLC(Map(index -> expected.confidence))
      (in, out)
    }}

    // Value that the new model instances will be filled with.
    val rng = new Random()
    val filler = () => rng.nextDouble * 4d - 2d

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
  def predict(modelInstance: ModelInstance, query: Query): Result = {
    import Model._
    import sentimentVec.traverseSyntax._

    // Parse the sentence into a tree.
    val tree = Parser(query.sentence)

    // Apply the model.
    val value = model(modelInstance.get)(tree.map(Constant(_)))

    // Extract the class with highest confidence.
    val prediction = value.toList.zipWithIndex.maxBy(_._1)
    Result(
      sentiment = sentiments.inverse(prediction._2),
      confidence = prediction._1
    )
  }

  /**
   * A mapping between sentiment vector indices and sentiment labels.
   */
  val sentiments = BiMap.stringInt(Array("No", "Yes", "NA"))
}
