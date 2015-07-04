Overview
======

This template implements various algorithms for sentiment analysis, most based on recursive neural networks (RNN) and recursive neural tensor networks (RNTN)[1]. It uses an experimental library called Composable Machine Learning (CML) and the Stanford Parser. The example data set is the Stanford Sentiment Treebank.

[1] Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." Proceedings of the conference on empirical methods in natural language processing (EMNLP). Vol. 1631. 2013.

Installation
======

First, you need to download and install a modified version of scalaz (with enabled serialization):
```bash
> git clone http://github.com/pawel-n/scalaz
> cd scalaz
> sbt publishLocal
```

Then, install CML:

```bash
> git clone http://github.com/pawel-n/cml
> cd cml
> sbt publishLocal
```

Next, get the template:
```bash
> pio template get template-scala-cml-sentiment <EngineDir>
> cd <EngineDir>
```

Problem description
=====

The task is to classify (English) sentences into five classes: Very negative, Negative, Neutral, Positive, Very positive. The input is either a sentence already parsed into a tree (when training) or raw text (in deployment). The output is a tree of *sentiment vectors*, i.e. for each part of the sentence the algorithm predicts a vector of probabilities - one value for each class.

Data Source
=====

The data source loads parsed sentence trees from files "data/train.txt" and "data/test.txt". You can set how much
data should be loaded by changing the "fraction" setting. Data is grouped into batches of size given by "batchSize".

Algorithms
=====

The template implements 5 algorithms:
* RNN - recursive neural network,
* RNTN - recursive neural tensor network,
* RNTNSimpl - a variation of RNTN with a simplier combining function,
* RNTNSplit - a variation of RNTN where the word vectors are split in 2 parts and combining is done using a (0, 4)-tensor,
* MM - a simple model based on matrix multiplication, every word is mapped to a matrix, matrices are multiplied and the result is classified with softmax regression.

Every algorithm shares the same cost function and optimization method (AdaGrad). The common functionality is provided in the AlgorithmBase class. The only thing missing in AlgorithmBase is a *model*, which is defined by a trait:
```scala
trait Model[In[_], Out[_]] extends Serializable {
  /**
   * The space of parameters.
   */
  type Params[A]

  /**
   * Parameters must form a representable vector space.
   */
  implicit val params: Representable[Params]

  /**
   * Applies the model to some input.
   * @param input The input.
   * @param params The model parameters.
   * @param a Number operations (like addition, multiplication, trigonometric functions).
   * @tparam A The number type.
   * @return The output.
   */
  def apply[A](params: Params[A])(input: In[A])(implicit a: Analytic[A]): Out[A]
}
```
Each model has an input type In[\_], output type Out[\_], a parameter space Params[\_]. If you have a parameter vector, you can apply a model to some input, yielding some output. In our case the input type is Tree\[Unit, String\] (a binary tree with strings in the leafs and no information in the nodes) and the output type is Tree\[SentimentVector, String\] (a binary tree with strings in the leafs and a probability distribution over a set of classes in the nodes).

Models can be implemented directly, however CML provides a library of basic models (linear functions, scalar functions applied pointwise, tensors, map/reduce). Furthermore, models can be composed, i.e. if we have a model going from A to B and another from B to C the composition will take the input of type A and yield output of type C.

CML uses automatic differentiation to compute the gradients require to optimize models.

Building
=====

Run the standard command:
```bash
> pio build
```

Evaluation
=====

The template evaluates the accuracy with 4 metrics:
* AccuracyRoot - checks what fraction of whole sentences (the tree roots) had their sentiment predicted correctly,
* AccuracyAll - checks what fraction of sentence fragments had their sentiment predicted correctly,
* AccuracyBinaryRoot - like AccuracyRoot, but only considers whether the sentiment is positive or negative (i.e. collapses Positive and Very Positive into a single class),
* AccuracyBinaryAll - like AccuracyAll, but only considers whether the sentiment is positive or negative.

To run the evaluation execute:
```bash
> SPARK_MEM="4g" pio eval -sk sentiment.SentimentEvaluation
```

SPARK_MEM controls the amount of memory given to the engine. Generally, the more the better.

Warning: if run on the entire data set (as is the default) this can take a very long time, up to 10 hours.

The evaluation results, compared to the Stanford implementation:

Name       | Vector/matrix size | All  | Roots | All binary | Root binary
---------- | ------------------ | ---- | ----- | ---------- | -----------
RNN        | 10                 | 79.4 | 48.7  | 85.0       | 76.6
RNTN       | 10                 | 76.7 | 37.7  | 82.5       | 64.1
RNTNSimple | 10                 | 78.6 | 45.6  | 84.3       | 75.0
RNTNSplit  | 10                 | 75.1 | 30.1  | 79.3       | 47.2
MM         | 15x15              | 76.2 | 6.0   | 78.5       | 10.6
Stanford RNTN | 25-35           | 80.7 | 45.7  | 87.6       | 85.4

The performance is very close on tests All, Root and Binary, however it is lower then expected on the RootBinary test. We have used smaller word vector sizes to save time. Increasing them to 25-35 should bring the performance on RootBinary up to Stanford results.

Training
====

To train the engine run:
```bash
> SPARK_MEM="4g" pio train
```

The default configuration uses only 10% of the available data.

You can choose the fraction of data used in engine.json and the used algorithm - just provide the configuration only for the one you want to use. Have a look at the SentimentEvaluation object in *Accuracy.scala* for some configuration examples.

Visualisation
=====

The template includes a very simple visualisation in the form of a Django app. To run it first install the requirements:
```bash
> cd <EngineDir>/web/
> virtualenv env
> source env/bin/activate
> pip install -r requirements.txt
```
Next, run the web server:
```bash
> ./manage.py runserver
```

Finally, deploy the engine (make sure it is trained!):
```bash
> cd <EngineDir>
> pio deploy --port 8001
```

The visualisation is available at http://localhost:8000/. A submitted sentence will be parsed into a tree and each node will be colored based on the sentiment of that sentence fragment. Red color denotes negative and green denotes positive sentiment.
