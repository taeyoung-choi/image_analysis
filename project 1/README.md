## Assignment 1: practice an image segmentation algorithm using K-means

##### TaeYoung Choi (tc2777)
##### Monday February 19, 2018

1. The parameter K was chosen from the histogram. The assumption is that the centroids will converge to the pixel 
values that appear often. If the number of clusters exceeds the number of peaks in the histogram, then it would make the 
algorithm unstable, because the extra clusters will appear in different locations each iteration. Thus it was 
reasonable to pick K according to the shape of histogram.
![histogram](https://github.com/taeyoung-choi/image_analysis/blob/master/data/hist.png)

2. After running K-means, it was necessary to check which color corresponds to the face. This was something manually 
coded and this can be automated using a smarter algorithm.
![k-coloring](https://github.com/taeyoung-choi/image_analysis/blob/master/data/k_color.png)

3. Face Detection: We'd like to find the best region that includes as many blue points (face pixels) as possible, but we do not want the entire image either. Hence, the metric I used is the following:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{total\&space;number\&space;of\&space;blue\&space;pixels}}\times&space;\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{size\&space;of\&space;the\&space;candidate\&space;region}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{total\&space;number\&space;of\&space;blue\&space;pixels}}\times&space;\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{size\&space;of\&space;the\&space;candidate\&space;region}}" title="\frac{\mathrm{number\ of\ blue\ pixels}}{\mathrm{total\ number\ of\ blue\ pixels}}\times \frac{\mathrm{number\ of\ blue\ pixels}}{\mathrm{size\ of\ the\ candidate\ region}}" /></a></p>

4. Result: This approach fails when there are multiple faces with different ranges of pixel values. The color information would not be the best measure for face detection.

![face](https://github.com/taeyoung-choi/image_analysis/blob/master/data/face_detected.png)

All variables and parameters are explained in the code.
