package sentiment

import cml._
import cml.algebra._
import cml.models._

case class MMParams (
  vecSize: Int,
  stepSize: Double,
  iterations: Int,
  regularizationCoeff: Double,
  noise: Double
) extends AlgorithmParams

class MM (
  params: MMParams
) extends AlgorithmBase (params) {
  // First declare the size of our vectors. We use RuntimeNat here because the size depend on algorithm parameters.
  val vecSize = algebra.RuntimeNat(params.vecSize)

  // Now lets declare the types of vectors that we'll be using.
  type Vect[A] = Vec[vecSize.Type, A]
  type Matrix[A] = Vect[Vect[A]]
  type MatrixPair[A] = (Matrix[A], Matrix[A])
  type MatrixTree[A] = Tree[Matrix[A], String]

  // We have to find the required implicits by hand because Scala doesn't support type classes.
  implicit val vecSpace = Vec.cartesian(vecSize())
  implicit val matrixMonoid = Monoid1.matrix[Vect]

  val model = Chain2[InputTree, MatrixTree, OutputTree](
    AccumulateTree[Word, Matrix](
      inject = SetMap[String, Matrix],
      reduce = Chain2[MatrixPair, Matrix, Matrix](
        MonoidAppend[Matrix],
        Pointwise[Matrix](AnalyticMap.tanh)
      )
    ) : Model[InputTree, MatrixTree],
    BifunctorMap[Tree, Matrix, Sentiment.Vector, Word, Word](
      left = Chain2(
        AffineMap[Matrix, Sentiment.Vector],
        Softmax[Sentiment.Vector]
      ),
      right = Identity[Word]
    )
  )
}
