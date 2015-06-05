package sentiment

import io.prediction.controller.{Params, PPreparator}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class Preparator extends PPreparator[TrainingData, TrainingData] {
  def prepare(sc: SparkContext, trainingData: TrainingData): TrainingData = {
    trainingData
  }
}