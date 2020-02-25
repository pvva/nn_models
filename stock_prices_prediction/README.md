In here I'm trying to predict ~~ripples of the water~~ future stock prices by history ones.

There are 2 models: linear (so so), rnn based (quite promising).

RNN.
- [Article](https://arxiv.org/pdf/1704.02971.pdf)
- Reference implementations: [one](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), [two](https://github.com/chandlerzuo/chandlerzuo.github.io/tree/master/codes/da_rnn), [three](https://github.com/Seanny123/da-rnn)

Data.
Data represents simple array of prices, ordered in time.

In order for the model to be able to work with data, it is combined into 2 dimensional matrix with following structure:

[

P[0]..P[T-1]

P[1]..P[T]

P[2]..P[T+1]

...

]

Here P[0], P[1] and so on are values in original prices array.

For training and inference batches of 2D matrices are formed, so that input has size of B x T-1 x I,
where B is batch size, T is time steps amount (how many continuous steps to take), I is single input data size
 (for example, if this model would be used for text, it would be an embedding vector size).
 
 All code in `model_rnn.py` is commented with specifying dimensions at every step, so that it's understandable for anyone.
 
 Key factor in the model is, of course, attention mechanism.
 
 Best prediction would anyway come from the input, that represents different commodities and/or currencies prices, and target stock/commodity price,
  so that model is using patterns in related sequences to learn target one.
  