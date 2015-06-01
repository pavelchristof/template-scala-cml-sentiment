package sentiment

import edu.stanford.nlp.trees.Tree
import io.prediction.controller.LPreparator

class Preparator extends LPreparator[TrainingData, PreparedData] {
  override def prepare(trainingData: TrainingData): PreparedData = {
    new PreparedData(sentences = trainingData.sentences.par.map { case (a, b) => (Parser(a.sentence), b) }.toArray)
  }
}

class PreparedData(
  val sentences: Array[(Tree, String)]
) extends Serializable
