package sentiment

import io.prediction.controller.LServing

class Serving extends LServing[Query, SentenceTree] {
  override def serve(query: Query,
    predictedResults: Seq[SentenceTree]): SentenceTree = {
    predictedResults.head
  }
}
