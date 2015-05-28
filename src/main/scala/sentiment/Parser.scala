package sentiment

import java.io.StringReader

import edu.stanford.nlp.parser.lexparser.LexicalizedParser
import edu.stanford.nlp.process.{LowercaseAndAmericanizeFunction, PTBTokenizer}
import edu.stanford.nlp.trees.{TreeTransformer, Tree}


object Parser {
  /**
   * Parser a sentence into a tree.
   */
  def apply(sentence: String): Tree = {
    val reader = new StringReader(sentence)
    val tokenizer = PTBTokenizer.newPTBTokenizer(reader, false, false)
    parser
      .parse(tokenizer.tokenize())
      .transform(Normalize)
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
}
