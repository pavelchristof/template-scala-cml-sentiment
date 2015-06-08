package sentiment

import cml._
import cml.algebra._
import cml.models._

case class RNTNParams (
  wordVecSize: Int,
  stepSize: Double,
  iterations: Int,
  regularizationCoeff: Double,
  noise: Double
) extends AlgorithmParams

class RNTN (
  params: RNTNParams
) extends AlgorithmBase (params) {
  // First declare the size of our vectors. We use RuntimeNat here because the size depend on algorithm parameters.
  val wordVecSize = algebra.RuntimeNat(params.wordVecSize)

  // Now lets declare the types of vectors that we'll be using.
  type WordVec[A] = Vec[wordVecSize.Type, A]
  type WordVecPair[A] = (WordVec[A], WordVec[A])
  type WordVecQuad[A] = (WordVecPair[A], WordVecPair[A])
  type WordVecTree[A] = Tree[WordVec[A], String]

  // We have to find the required implicits by hand because Scala doesn't support type classes.
  implicit val wordVecSpace = Vec.cartesian(wordVecSize())
  implicit val wordVecPairSpace = Cartesian.product[WordVec, WordVec]
  implicit val wordVecQuadSpace = Cartesian.product[WordVecPair, WordVecPair]

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
