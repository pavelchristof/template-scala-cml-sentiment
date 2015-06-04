package sentiment

import io.prediction.controller.LServing

class Serving extends LServing[Query, Result] {
  override def serve(query: Query,
    predictedResults: Seq[Result]): Result = {
    predictedResults.head
  }
}
