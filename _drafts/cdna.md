---
layout: post
title: "Unsupervised Learning for Physical Interaction through Video Prediction"
comments: true
excerpt: By Chelsea Finn, Ian Goodfellow, and Sergey Levine. NIPS, 2016
---

**TL'DR:**
Instead of trying to directly predict the pixel values for the next frame or even the difference from the previous frame, the authors propose 3 models (DNA, CDNA, STP) that predict the motion of each image part and apply it to the corresponding region of the last frame (by masking out the others with predicted masks), since appearance information is available therein. Consequently, the models learn to distangle appearance from motion and are (partially) invariant to appearance, being able to predict multiple timesteps ahead.

---

**Models:**
---

{% include image.html file="finn-cdna.png" alt="CDNA ar chitecture"
description="On the left we see the proposed general framework-agnostic architecture and on the right, the deep learning implementation studied." %}

---

**Useful links:**
---

[GitHub](https://github.com/tensorflow/models/tree/master/research/video_prediction)
[Project page](https://sites.google.com/site/robotprediction/)

**References**
---

[1]

[2]