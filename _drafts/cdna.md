---
layout: post
title: "Unsupervised Learning for Physical Interaction through Video Prediction"
comments: true
excerpt: By Chelsea Finn, Ian Goodfellow, and Sergey Levine. NIPS, 2016
---

**TL'DR:**
Instead of trying to directly predict the pixel values for the next frame or even the difference from the previous frame, the authors propose 3 models (DNA, CDNA, STP) that predict the motion of each image part and apply it to the corresponding region of the last frame (by masking out the others with predicted masks), since appearance information is available therein. Consequently, the models learn to disentangle appearance from motion and are (partially) invariant to appearance, being able to predict multiple timesteps ahead.

---

**Models**
---

According to [Wikipedia](https://en.wikipedia.org/wiki/Advection), "advection is the transport of a substance by bulk motion. The properties of that substance are carried with it. ".

I sincerely didn't know the meaning of this word until reading this paper, so that's where *my* learning started. My opinion about this choice of word is that the authors wished to express that the objects (visual) properties (i.e.: color, shape, texture) are preserverd during the (predicted) motion.

Describe general features of the base architecture: convLSTM , skip-connections, encoder-decoder structure

"In an interactive setting, the agent’s actions and internal state (such as the pose of the robot gripper) influence the next image and so we integrate both into our model. Note, though, that the agent’s internal state (i.e. the robot gripper pose) is only input into the network at the beginning, and must be predicted from the actions in future timestep"

### DNA -- Dynamic Neural Advection

In this approach, we predict a distribution over locations in the previous frame for each pixel in the new frame.  The predicted pixel value is computed as an expectation under this distribution.  We constrain the pixel movement to a local region, under the regularizing assumption that pixels will not move large distances. This keeps the dimensionality of the prediction low. This approach is the most flexible of the proposed approaches. Formally, we apply the predicted motion transformation^mto the previous image prediction

Some description here

### CDNA -- Convolutional DNA

Further assuming that different regions of the image may present the same motion (e.g., a car on a street), the kernel weights of the motion matrices are shared throughout the whole image. Then, it suffices to predict only a few (i.e., 10) matrices and their soft masks, instead of $H \times W$ kernels.
Stiching it all together is pretty straightforward: it's the pointwise sum of each predicted image (computed by convolving a motion matrix by the input frame) weighted by the corresponding mask

For the example in the paper, this modification reduces in nearly 90% the number of parameters produced by the network.

{% include image.html file="finn-cdna.png" alt="CDNA architecture"
description="On the left we see the proposed general framework-agnostic architecture and on the right, the deep learning implementation studied." %}

### STP -- Spatial Transformer Network

Instead of directly predicting values for the motion matrices as CDNA, the output of the fully-connected layer is fed to a Spatial Transformer Netork (STN) [ref number]. Hence, this model's name. More specifically, " a set of affine parameters $\hat{M}$ produces a warping grid between previous image pixels(xt)"

**Experiments**
---

"We trained the networks using an l2 reconstruction loss."

"experiments show that the CDNA and STP models learn to mask out objects that are moving in consistent direction". "The DNA model on the other hand is more flexible as it can produce independent motions for every pixel in the image."



<figure class="image-wrapper">
<img src="https://storage.googleapis.com/robotpush-gifs/novel/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs0/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs11/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs10/2_96.gif" width="180" height="180" border="0"><br>
<figcaption>Qualitative comparison across models when predicting novel objects. (source: <a href="https://sites.google.com/site/robotprediction/">Project page</a>)<br>
From left to right: groundtruth, CDNA, ConvLSTM with skip, FF multiscale [14], FC LSTM [17]</figcaption>
</figure>

Description

<figure class="image-wrapper">
<img src="https://storage.googleapis.com/push_gens/novelgengifs9/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/mask0_2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/mask2_2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/mask8_2_96.gif" width="180" height="180" border="0"><br>
<figcaption>CDNA masks for predicted motions. From left to right: masks #0 (background motion), #2 and #8. (source:  <a href="https://sites.google.com/site/robotprediction/">Project page</a>)</figcaption>
</figure>

---

**Useful links:**
---

[GitHub](https://github.com/tensorflow/models/tree/master/research/video_prediction)

[Project page](https://sites.google.com/site/robotprediction/)

**References**
---

[1]

[2]