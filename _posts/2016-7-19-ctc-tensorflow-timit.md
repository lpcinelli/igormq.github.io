---
layout: post
title: CTC + Tensorflow + TIMIT
comments: true
excerpt: How to get the CTC working.
---

Before starting, I think that it’s better give a brief description about the dataset that I've been using. If you don't want to wait for the entire post, you can skip this and access the [GitHub code](https://github.com/igormq/ctc_tensorflow_example).

## The TIMIT dataset

TIMIT ([LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1)) is a speech dataset that was developed by Texas Instruments and MIT (hence the corpus name) with DARPA’s (Defense Advanced Research Projects Agency) financial support at the end of 80’s. This dataset has many applications, such as the study of acoustic and phonetic properties and the evaluation/training of automatic speech recognition systems (ASR).

There are broadband recordings of 630 speakers of eight major dialects of American English, each reading ten phonetically rich sentences.

<div class="table-wrapper center" markdown="block">

Region[^f1]|Men|Women|Total     
---|---|---|---|
 1|31 (63%)|18 (27%)|49 (8%)
 2|71 (70%)|31 (30%)|102 (16%)
 3|79 (67%)|23 (23%)|102 (16%)
 4|69 (69%)|31 (31%)|100 (16%)
 5|62 (63%)|36 (37%)|98 (16%)
 6|30 (65%)|16 (35%)|46 (7%)  
 7|74 (74%)|26 (26%)|100 (16%)
 8|22 (67%)|11 (33%)|33 (5%)
 ===|===|===|===|
 |438 (70%) |192 (30%) |630 (100%)

</div>

Each utterance is separated by three major categories: SA (dialect sentence), SX (compact sentence), and SI (diverse sentence).

The SA sentences were meant to show the dialectal variants of the speakers and were read by all 630 speakers. So, for an automatic speaker independent recognition system, **these sentences must be ignored**.

The phonetically-compact (SX) sentences were designed to provide a good coverage of pairs of phones, with extra occurrences of phonetic contexts thought to be either difficult or of particular interest. Each speaker read 5 of these sentences and each text was spoken by 7 different speakers.

Finally, the phonetically-diverse (SI) sentences were selected from existing text sources, such as Brown corpus[^2] and Playground dialogs, so as to add diversity in sentences types. Each speaker read 3 of these sentences, with each sentence being read only by one speaker.

All audio files were recorded in a controlled environment, as you can hear in the following example:

<div class="table-wrapper center" markdown="block">
<audio controls>
  <source src="https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>
</div>


#### Division

Each utterance in the TIMIT dataset has its own time-aligned orthographic (.txt files), phonetic (.phn files) and word transcriptions (.wrd files) as well as 16-bit, 16kHz speech waveform in the NIST format (.wav files). Also, the dataset is separated by two major folders: test and train. The test set has 168 speakers and 1344 utterances available (regarding that SA sentences weren’t meant to be used in ASR systems). This test set is also called as complete test set.

##### *Core test set*

Using the complete test set has a drawback: the intersection of SX sentences by speakers. Facing that, the researches would rather evaluate the ASR system in the core test set.

The core test set has 24 speakers, 2 men and 1 woman of each dialect region, where each one read 5 unique SX sentences plus its 3 SI sentences, given 192 utterances.

<div class="table-wrapper center" markdown="block">

Region | Man | Woman
---|---|---
1|DAB0, WBT0|ELC0
2|TAS1, WEW0|PAS0
3|JMP0, LNT0|PKT0
4|LLL0, TLS0|JLM0    
5|BPM0, KLT0|NLP0
6|CMJ0, JDH0|MGD0
7|GRT0, NJM0|DHC0
8|JLN0, PAM0|MLD0

</div>

where the column `Man` and `Woman` shows the unique speaker identification.

### More information
For more information, please access the [LDC](https://catalog.ldc.upenn.edu/LDC93S1) website or read the [online documentation](https://catalog.ldc.upenn.edu/docs/LDC93S1/)  of TIMIT dataset.


## Tensorflow

At the beginning of this post you should have asked: why are you using Tensorflow for recurrent networks?

Before began this project, @robertomest and I have searched for several toolkits that have been developed by many researches in the Deep Learning area. Among of them, there are [Torch](http://torch.ch), [Theano](http://deeplearning.net/software/theano/), [PyLearn2](https://github.com/lisa-lab/pylearn2), [Caffe](http://caffe.berkeleyvision.org), [CNTK](https://www.cntk.ai), and [Tensorflow](https://www.tensorflow.org) as the top toolkits.

Each tool has its own advantages and disadvantages (see [here](https://github.com/zer0n/deepframeworks) the comparisons), and after all, we chose one that had great coverage for image and speech applications and wasn’t developed to be used with pipelines or configurations files (like CNTK). Plus, we’ve already had some familiarities with Python. So, we chose Tensorflow, and when it’s possible, we use the [Keras](http://keras.io) frontend.

### Installing

My setup is quite simple. Seriously. If you choose the correct tool for installing Python, you will not have any headache. The first step is installing Python through the Anaconda (developed by Continuum), so the next steps will be the same in all OS systems.

#### Anaconda

Anaconda can be found [here](https://www.continuum.io/downloads) and you can choose installing from python 2.7 or 3.5, but not necessarily it will be the version that you will use for development, because Anaconda works like virtualenv, but it’s way better.

#### Creating the environment

After installing the Anaconda e add the `anaconda/bin` directory in your system’s path, you have to create the environment and install some packages.

```bash
conda create –n tensorflow python=2.7 matplotlib pillow scipy scikit-learn scipy numpy h5py jupyter
```
The command ```coda create –n tensorflow``` will create a new environment with the name tensorflow and the option ```python=2.7``` will install python version 2.7. After that, all names are packages that will be installed and are necessary for the development.

##### Activating the environment

Finally, all you have to do is write

```bash
source activate tensorflow
```
to activate the tensorflow environment.

#### Installing Tensorflow

Installing the Tensorflow is as easily as installing Anaconda. You can follow the step-by-step tutorial [here](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#anaconda-installation).

## CTC

Instead of using DNN-HMM approaches for ASR systems, I will follow another line of research: end-to-end speech recognition. There are two major areas: using RNN networks with custom cost function, the Connectionist Temporal Classification[^3] (CTC) or using an encoder-decoder system with attention[^4].

For now, I will focus on systems using the CTC cost function. In this post, I will not explain how CTC works, and I will delay this task for other post. Sorry for that.

### The short introduction

CTC is a cost function used for tasks where you have variable length input and variable length output and you don’t know the alignment between them. Hence, for the TIMIT task, we will not use the time-alignment of transcriptions, because the CTC can automatically find these alignments.

## Coding!

Now that I made a short introduction (or not so short; sorry about that) we will start coding.

CTC has already been implemented in Tensorflow since version 0.8 in [`contrib`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/ctc_ops.py#L30) module (yey!), but is quite confusing using it for the first time. The python docstring isn’t helpful and the solution is going deep and read the docstring in the [.c file](https://github.com/tensorflow/tensorflow/blob/d42facc3cc9611f0c9722c81551a7404a0bd3f6b/tensorflow/core/ops/ctc_ops.cc#L32) and read the [test scripts](https://github.com/tensorflow/tensorflow/blob/679f95e9d8d538c3c02c0da45606bab22a71420e/tensorflow/python/kernel_tests/ctc_loss_op_test.py) from Tensorflow’s GitHub page. Fortunately (or not), I will try to explain better how we can use this function!

### Vanilla system

The vanilla system consists basically of one or more layers of recurrent neural networks (RNN, LSTM, GRU, and so on) followed by affine transformation, softmax layer and the loss will be evaluated through CTC function.

#### Dealing with the input

As input of computational graph, we have the utterances, the targets, and the  sequence length of each utterance (for dynamic unroll of RNN).

```python
import tensorflow as tf

# e.g: log filter bank or MFCC features
# Has size [batch_size, max_stepsize, num_features], but the
# batch_size and max_stepsize can vary along each step
inputs  = tf.placeholder(tf.float32, [None, None, num_features])

# Here we use sparse_placeholder that will generate a
# SparseTensor required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32)

# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])
```

One of the news in this code is the `sparse_placeholder` for the targets. You can feed this placeholder in the following ways:

```python
session = tf.Session()

# Graph definition
x = tf.sparse_placeholder(tf.float32)
y = tf.sparse_reduce_sum(x)

# Values to feed the sparse placeholder
indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
values = np.array([1.0, 2.0], dtype=np.float32)
shape = np.array([7, 9, 2], dtype=np.int64)

# Option 1
print(sess.run(y, feed_dict={ x: tf.SparseTensorValue(indices, values, shape)}))  

# Option 2
print(sess.run(y, feed_dict={x: (indices, values, shape)}))

# Option 3
sp = tf.SparseTensor(indices=indices, values=values, shape=shape)
sp_value = sp.eval(session)

print(sess.run(y, feed_dict={x: sp_value}))

session.close()
```

#### Recurrent network
Given the input, now we can feed our network and calculate all the states, so

```python
# Defining the cell
# Can be:
#   tf.nn.rnn_cell.RNNCell
#   tf.nn.rnn_cell.GRUCell
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

# Stacking rnn cells
stack = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,
                                    state_is_tuple=True)

# The second output is the last state and we will no use that
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
```

The argument `state_is_tuple` is set to `True` to avoid annoying warning (and will be the default in the next versions of Tensorflow). The code presented above is very straightforward.

Using `tf.nn.rnn` instead of `tf.nn.dynamic_rnn` has severals drawbacks. `tf.nn.rnn` cannot performs dynamic unroll of the network, making the graph growing when the time step is big, allocating a lot of memory and slowing the forward/backward pass. Furthermore, `tf.nn.rnn` can only be used if the time step is the same across all batches. For now on, we'll only use the dynamic rnn function.

Here, we use only the directional network, but the results can be improved if we use a bidirectional mode (only in the master version on Tensorflow).

#### Fully Connected

The next step is to apply at each time step one fully connected network, sharing the weights over time. First of all, we need to reshape our output

```python
batch_size, max_timesteps = tf.shape(inputs)[:2]
outputs = tf.reshape(outputs, [-1, num_hidden])
```

After that, we will apply the affine transformation

```python
# Truncated normal with mean 0 and stdev=0.1
# Tip: Try another initialization
# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
W = tf.Variable(tf.truncated_normal([num_hidden,
                                     num_classes],
                                    stddev=0.1))
# Zero initialization
# Tip: tf.zeros_initializer
b = tf.Variable(tf.constant(0, shape=[num_classes]))

# Doing the affine projection
logits = tf.matmul(outputs, W) + b

# Reshaping back to the original shape
logits = tf.reshape(logits, [batch_size, -1, num_classes])
```

where the `tf.truncated_normal` and `tf.constant` are initializers and `num_classes` will be `num_labels + 1` where the last class is reserved to the blank label.

#### Softmax and CTC loss

Here, we need an extra attention. The CTC loss automatically performs the `softmax` operation, so we can skip this operation. Also, the CTC requires an input of shape `[max_timesteps, batch_size, num_classes]` (and I don’t know why, because the Tensoflow's code isn't time major by default).

```python
# Time major
logits = tf.transpose(logits, (1, 0, 2))

loss = tf.contrib.ctc.ctc_loss(logits, targets, seq_len)
cost = tf.reduce_mean(loss)
```

#### Accuracy
To evaluate our system, we can use one of two decoders available at `tf.contrib.ctc` module

```python
# Option 2: tf.contrib.ctc.ctc_beam_search_decoder
# (it's slower but you'll get better results)
decoded, log_prob = tf.contrib.ctc.ctc_greedy_decoder(logits, seq_len)

# Accuracy: label error rate
acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                      targets))
```

#### Packing everything and running
"Is it only this?". Yes, it is. But, don't think that the hard working is there, specially after I spent hours trying to understand every parameter and the tricky docstring. You have still to pre-process the dataset in the right way, feed the placeholders, choose the optimizer, fine tuning the hyper parameters and yet will be hard to get some results found by other researchers.

##### Gist code!

You can find a working implementation on [GitHub](https://github.com/igormq/ctc_tensorflow_example)! Feel free to use.

## I haven't finished yet
There is more! In the next post, I'll show you my efforts trying to get the same LER in the Grave's PhD Thesis[^1].

*See you soon!*

## References & Footnotes

[^f1]: Dialect regions: (1) New England, (2) Northern, (3) North Midland, (4) South Midland, (5) Southern, (6) New York City, (7) Western, (8) Army Brat (moved around)
[^1]: Graves, Alex. "Neural Networks." Supervised Sequence Labelling with Recurrent Neural Networks. Springer Berlin Heidelberg, 2012.
[^2]: Francis, W. Nelson, and Henry Kucera. "Brown corpus manual." Brown University (1979).
[^3]: Graves, Alex, et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." Proceedings of the 23rd international conference on Machine learning. ACM, 2006.
[^4]: Chorowski, Jan, et al. "End-to-end continuous speech recognition using attention-based recurrent NN: first results." arXiv preprint arXiv:1412.1602 (2014).
