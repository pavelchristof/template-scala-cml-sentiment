package sentiment

import edu.stanford.nlp.trees.Tree
import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(sentences = trainingData.sentences.map { case (a, b) => (Parser(a.sentence), b) })
  }
}

class PreparedData(
  val sentences: Array[(Tree, String)]
) extends Serializable