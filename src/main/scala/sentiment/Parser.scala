package sentiment

import scalaz.Tree

object Parser {
  /**
   * Parser a sentence into a tree.
   */
  def apply(sentence: String): Tree[String] =
    toTree(sentence.trim.split(" ").map(_.filter(_.isLetterOrDigit).map(_.toLower)))

  def toTree(seq: Seq[String]): Tree[String] =
    seq match {
      case Seq() => Tree.leaf("STOP")
      case x +: xs => Tree.node(x, Stream(toTree(xs)))
    }
}
