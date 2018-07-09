---
layout: post
title: "Unsupervised Learning for Physical Interaction through Video Prediction"
comments: true
use_math: true
excerpt: By Chelsea Finn, Ian Goodfellow, and Sergey Levine. NIPS, 2016
---

**TL'DR:**

Instead of trying to directly predict the pixel values for the next frame or even the difference from the previous frame, the authors propose 3 models (DNA, CDNA, STP) that predict the motion of each image part and apply it to the corresponding region of the last frame (by masking out the others with predicted masks), since appearance information is available therein. Consequently, the models learn to disentangle appearance from motion and are (partially) invariant to appearance, being able to predict multiple timesteps ahead.

**Models**
---

According to [Wikipedia](https://en.wikipedia.org/wiki/Advection), "advection is the transport of a substance by bulk motion. The properties of that substance are carried with it. ".

I sincerely didn't know the meaning of this word until reading this paper, so that's where *my* learning started. My opinion about this choice of word is that the authors wished to express that the objects (visual) properties (i.e.: color, shape, texture) are preserverd during the (predicted) motion.

Describe general features of the base architecture: convLSTM , skip-connections, encoder-decoder structure

"In an interactive setting, the agent’s actions and internal state (such as the pose of the robot gripper) influence the next image and so we integrate both into our model. Note, though, that the agent’s internal state (i.e. the robot gripper pose) is only input into the network at the beginning, and must be predicted from the actions in future timestep"

### DNA -- Dynamic Neural Advection

The model generates one motion matrix for each pixel of the frame. Thus, the it computes the prediction for each new pixel value as the dot product between the corresponding motion matrix and the region around such pixel. The authors explain the motion matrix as a distribution over locations and the new pixel value as the expected value of the distribution, this is an interesting (and cool) view. Wouldn't you rather say your model outputs a distribution over locations whose expectation will be the new value than say it outputs an ordinary matrix and computes the dot product?

Assuming spatial locality over the frames across each timestep, and that pixels will not move large distances, the size of such motion matrices can be small, which considerably reduces the problem's dimension.

### CDNA -- Convolutional DNA

Further assuming that different regions of the image may present the same motion (e.g., a car on a street), the per pixel matrix can be replaced by kernels whose weights are shared throughout the whole image. Then, it suffices to predict only a few (i.e., 10) matrices, instead of $H \times W$. Additionally, the model generates one (soft) mask per motion matrix so that it can stitch everything together by selecting how much of each motion matrix goes into the prediction of each pixel.

The authors also include one extra "background" mask which simply copies the values of the previous frame into the prediction. Such behavior typically occurs for background regions where there is no movement. Hence, the name.

This approach reduces in nearly 90% the number of values produced by the network.

{% include image.html file="finn-cdna.png" alt="CDNA architecture"
description="On the left we see the proposed general framework-agnostic architecture and on the right, the deep learning implementation studied." %}

### STP -- Spatial Transformer Predictors

Instead of producing values for the motion matrices as CDNA, this model outputs a set of parameters that are fed to a Spatial Transformer Network (STN) [ref number]. Hence, the model's name.

The transformations are modelled as 2D affine and aim to (affinely) warp the previous image pixels onto the future pixels.

**Experiments**
---

The authors introduce a new dataset, named BAIR Robot Pushing dataset, consisting of sequences of robotic arms pushing different objects in bins, the corresponding gripper poses (internal states) and their commands (actions). Besides this new dataset, the authors also evaluate the proposed architectures on the Human3.6M dataset [ref], which consists of humans performing various actions in a room.

Finn and collaborators verify that modeling pixel motion outperforms directly predicting pixel values or even the difference between previous and current frames. Not only that but such gap enlarges when dealing with novel objects. Furthermore, training for longer horizons and conditioning on the action/state of the robot enhance performance.

<figure class="image-wrapper">
<img src="https://storage.googleapis.com/robotpush-gifs/novel/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs0/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs11/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs10/2_96.gif" width="180" height="180" border="0"><br>
<figcaption>Qualitative comparison across models when predicting novel objects. (source: <a href="https://sites.google.com/site/robotprediction/">Project page</a>)<br>
From left to right: groundtruth, CDNA, ConvLSTM with skip, FF multiscale [14], FC LSTM [17]</figcaption>
</figure>

Their models also surpass prior methods on Human3.6M. Nevertheless, after 10 timesteps (amount for which it was trained to predict), it starts to degrade. Note that each timestep corresponds to 1 second due to subsampling.

Naturally, prediction degrades over time due to future's inherent uncertainty. As the models are trained with $\ell_2$ norm, this translates to blur. If the model thinks there's a 50/50 chance that the object is going to be either in position A or B, then it depicts the object in both position simultaneously but halves the pixel intensity of each.

Although the DNA model is more flexible, the CDNA and STP ones are more efficient. Indeed, the latter architectures learn to mask out objects that are moving in consistent direction as shown below.

<figure class="image-wrapper">
<img src="https://storage.googleapis.com/push_gens/novelgengifs9/2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/mask0_2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/mask2_2_96.gif" width="180" height="180" border="0">&nbsp;<img src="https://storage.googleapis.com/push_gens/novelgengifs9/mask8_2_96.gif" width="180" height="180" border="0"><br>
<figcaption>CDNA masks for predicted motions of novel objects. From left to right: masks #0 (background motion), #2 and #8. (source:  <a href="https://sites.google.com/site/robotprediction/">Project page</a>)</figcaption>
</figure>

All three models have similar results on the robot pushing dataset, specially when dealing wiith novel objects. However, for the held-out subject in Human3.6M, the CDNA has worse performance than the other two.

### Observations

Since prediction comes from applying a transformation directly to the present frame, the model is fated to get image boundaries incorrectly. Moreover, if the object part is occluded, it cannot be predicted by combining nearby pixels. In order to address this issue, the authors (in the CDNA and STP models) generate raw pixel value predictions from the last LSTM cell, which is stitched together with the other predictions by its corresponding mask.

An interesting way to handle occlusions, at least for known objects, is by introducing a long range feedback skip connection. Thus, allowing long term memory and higher level concepts back to bottom layers, which may help to predict occluded or even not yet seen object parts of known objects.

It seems to me, by inspecting the code, that the CDNA approach runs short of a mask at the end, since it generates 10 different transformation matrices plus the generated raw pixel values while it outputs only 11 masks and 1 is already used as the background mask (i.e., it copies the values from the previous frame).

---

**Useful links:**
---

[GitHub](https://github.com/tensorflow/models/tree/master/research/video_prediction)

[Project page](https://sites.google.com/site/robotprediction/)

[BAIR Robot Pushing Dataset](http://sites.google.com/site/robotprediction)

[Human3.6M](http://vision.imar.ro/human3.6m/description.php)

**References**
---

[1]

[2]

[3] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments.PAMI , 36(7), 2014