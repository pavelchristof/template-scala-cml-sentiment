package sentiment

import java.io.StringReader

import edu.stanford.nlp.parser.lexparser.LexicalizedParser
import edu.stanford.nlp.process.{LowercaseAndAmericanizeFunction, PTBTokenizer}
import edu.stanford.nlp.trees.{TreeTransformer, Tree}
import scalaz.Scalaz._

object Parser {
  /**
   * Parser a sentence into a tree.
   */
  def apply(sentence: String): cml.Tree[Unit, String] = {
    val reader = new StringReader(sentence)
    val tokenizer = PTBTokenizer.newPTBTokenizer(reader, false, false)
    val tree = parser
      .parse(tokenizer.tokenize())
      .transform(Normalize)
    binarize(tree)
  }

  val parserModel = "data/englishPCFG.caseless.ser.gz"
  val parser = LexicalizedParser.loadModel(parserModel)

  object Normalize extends TreeTransformer {
    val normalize = new LowercaseAndAmericanizeFunction()

    override def transformTree(t: Tree): Tree = {
      if (t.isLeaf) {
        t.setValue(normalize(t.value()))
      }
      t
    }
  }

  def binarize(t: Tree): cml.Tree[Unit, String] = {
    if (t.isLeaf) {
      cml.Leaf((), t.value())
    } else {
      t.children().map(binarize).toVector.foldr1Opt(l => r => cml.Node(l, (), r)).get
    }
  }
}
