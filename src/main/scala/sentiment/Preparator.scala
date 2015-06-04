package sentiment

import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext

class Preparator extends PPreparator[TrainingData, TrainingData] {
  def prepare(sc: SparkContext, trainingData: TrainingData): TrainingData = {
    trainingData
  }
}
