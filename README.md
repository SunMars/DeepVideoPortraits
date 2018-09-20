# Rebuild DeepVideoPortrait

## Basic info

This is a rebuilding of [DeepVideoPortrait](https://web.stanford.edu/~zollhoef/papers/SG2018_DeepVideo/page.html).
We hope to realise almost the same output with theirs, based on which there may lie a choice to get some promote and learn something with the same time.

## Work progress

Thanks to the work of [3DMM_CNN](https://github.com/anhttran/3dmm_cnn), this project relies on their work to parameterize the face, including **identity** (shape&texture) and **expression**. And this project did some slightly change by transforming parameters of one graph to another, then render it.
As for the correspondence photo, just apply every pixel on the image with a RGB value according to the different index. And using opencv and the feature points detected in [3DMM_CNN](https://github.com/anhttran/3dmm_cnn), we target the eye region, which are painted to white. In this region, then, find the pupil next simply by spotting the darkest pixel in it and draw a blue circle with fixed radius centering that pixel.
In these ways, we now obtain the 3 images of every frame which applied as the conditioning inputs for cGAN in that paper.

## To do list

+ prepare the dataset and process frames in the videos first to gain that 3 images.
+ learing to build the structure of cGAN.
+ try to distribute this project on GPU and accelerate the training.
