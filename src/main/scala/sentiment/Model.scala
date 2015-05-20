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
import cml.algebra.Real._
import cml.algebra.Constant

object Model {
  import cml.models._
  import shapeless.Nat

  type Word[A] = Constant[String, A]
  implicit val word = Constant.concrete[String]
  implicit val strings = Enumerate.string(Enumerate.char)
  implicit val wordVector = algebra.Vector(Nat(5))
  implicit val wordVectorPair = algebra.Product[wordVector.Type, wordVector.Type]
  implicit val wordTree = algebra.Compose[Tree, Word]
  implicit val vectorTree = algebra.Compose[Tree, wordVector.Type]
  implicit val output = algebra.Vector(Nat(3))

  val model: cml.Model[wordTree.Type, output.Type] = Chain4(
    FunctorMap[Tree, Word, wordVector.Type](
      Function[String, wordVector.Type]
    ) : cml.Model[wordTree.Type, vectorTree.Type],
    Reduce[Tree, wordVector.Type](Chain2(
      AffineMap[wordVectorPair.Type, wordVector.Type],
      Pointwise[wordVector.Type](AnalyticMap.tanh)
    )) : cml.Model[vectorTree.Type, wordVector.Type],
    AffineMap[wordVector.Type, output.Type],
    Softmax[output.Type]
  )

  val costFun = new CostFun[wordTree.Type, output.Type] {
    override def scoreSample[A](sample: ScoredSample[wordTree.Type[A], output.Type[A]])
        (implicit an: Analytic[A]): A = {
      import an.analyticSyntax._
      val eps = fromDouble(0.0001)
      val j = ^(sample.expected, sample.actual){ case (e, a) => {
        - (e * (a + eps).log + (_1 - e) * (_1 - a + eps).log)
      }}
      output.sum(j)
    }

    override def regularization[V[_], A](instance: V[A])(implicit an: Analytic[A], space: LocallyConcrete[V]): A = {
      import an.analyticSyntax._
      fromDouble(0.01) * space.quadrance(instance)
    }
  }
}

case class Model (
  instance: Model.model.Type[Double]
)

class Algorithm extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    import cml.optimization._
    import Model._

    // Convert the data to our input format.
    val dataSet = data.sentences.map{ case (tree, expected) => {
      val index = sentiments(expected.sentiment)
      val in: wordTree.Type[Double] = tree.map(Constant(_))
      val out: output.Type[Double] = output.tabulateLC(Map(index -> expected.confidence))
      (in, out)
    }}

    val optimizer = GradientDescent(
      model,
      iterations = 100,
      step = 0.1
    )(wordTree.functor, output)

    val rng = new Random()
    val insts = for (i <- 0 until 1000)
      yield model.fill(rng.nextDouble() * 2d - 1d).asInstanceOf[optimizer.model.Type[Double]]
    //println(insts)
    implicit val diffEngine = ad.Forward
    optimizer[Double](insts.toVector, dataSet, costFun) match {
      case Vector(inst) => {
        println(inst)
        val i = inst.asInstanceOf[model.Type[Double]]
        for (t <- dataSet)
          println(model[Double](i)(t._1))
        Model(i)
      }
    }
  }

  def predict(model: Model, query: Query): PredictedResult = {
    import Model.output.traverseSyntax._

    val tree = Parser(query.sentence)
    val value = Model.model(model.instance)(tree.map(Constant(_)))

    val prediction = value.toList.zipWithIndex.maxBy(_._1)
    PredictedResult(
      sentiment = sentiments.inverse(prediction._2),
      confidence = prediction._1
    )
  }

  val sentiments = BiMap.stringInt(Array("No", "Yes", "NA"))
}
