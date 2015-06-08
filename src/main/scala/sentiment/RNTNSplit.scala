package sentiment

import cml._
import cml.algebra._
import cml.models._

case class RNTNSplitParams (
  halfVecSize: Int,
  stepSize: Double,
  iterations: Int,
  regularizationCoeff: Double,
  noise: Double
) extends AlgorithmParams

class RNTNSplit (
  params: RNTNSplitParams
) extends AlgorithmBase (params) {
  // First declare the size of our vectors. We use RuntimeNat here because the size depend on algorithm parameters.
  val halfVecSize = algebra.RuntimeNat(params.halfVecSize)

  // Now lets declare the types of vectors that we'll be using.
  type HalfVec[A] = Vec[halfVecSize.Type, A]
  type WordVec[A] = (HalfVec[A], HalfVec[A])
  type WordVecPair[A] = (WordVec[A], WordVec[A])
  type WordVecTree[A] = Tree[WordVec[A], String]

  // We have to find the required implicits by hand because Scala doesn't support type classes.
  implicit val halfVecSpace = Vec.cartesian(halfVecSize())
  implicit val wordVecSpace = Cartesian.product[HalfVec, HalfVec]

  object Tensor extends Model[WordVecPair, WordVec] {
    // Scala can't see the implicits that are right there /\
    val f4 = AffineMap[HalfVec, WordVec]()(halfVecSpace, wordVecSpace)
    val f3 = AffineMap[HalfVec, f4.Type]()(halfVecSpace, f4.space)
    val f2 = AffineMap[HalfVec, f3.Type]()(halfVecSpace, f3.space)
    val f1 = AffineMap[HalfVec, f2.Type]()(halfVecSpace, f2.space)

    override type Type[A] = AffineMap[HalfVec, f2.Type]#Type[A]
    override implicit val space = f1.space

    override def apply[A](inst: Type[A])(in: WordVecPair[A])(implicit a: Analytic[A]): WordVec[A] = {
      f4(f3(f2(f1(inst)(in._1._1))(in._1._2))(in._2._1))(in._2._2)
    }
  }

  override val model = Chain2[InputTree, WordVecTree, OutputTree](
    // In the first part of the algorithm we map each word to a vector and then propagate
    // the vectors up the tree using a merge function.
    AccumulateTree[Word, WordVec](
      // The function that maps words to vectors.
      inject = SetMap[String, WordVec],
      // Merge function, taking a pair of vectors and returning a single vector.
      reduce = Chain2(
        Tensor,
        Pointwise[WordVec](AnalyticMap.tanh)
      )
    ) : Model[InputTree, WordVecTree],

    // In the second part we map over the tree to classify the word vectors.
    BifunctorMap[Tree, WordVec, Sentiment.Vector, Word, Word](
      // Word vectors go thought a softmax classifier.
      left = Chain2(
        AffineMap[WordVec, Sentiment.Vector],
        Softmax[Sentiment.Vector]
      ),
      // Words are unchanged.
      right = Identity[Word]
    )
  )
}
