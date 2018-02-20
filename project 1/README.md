## Assignment 1: practice an image segmentation algorithm using K-means

##### TaeYoung Choi (tc2777)
##### Monday February 19, 2018

1. The parameter K was chosen from the histogram. The assumption is that the centroids will converge to the pixel 
values that appear often. If the number of clusters exceeds the number of peaks in the histogram would be the 
algorithm unstable, because the extra clusters will appear in different locations each iteration. Thus it was 
reasonable to pick K according to the shape of histogram.

2. After running K-means, it was necessary to check which color corresponds to the face. This was something manually 
coded and this can be automated using a smarter algorithm.

3. Face Detection: The region has to be big enough to fit the entire face and small enough to exclude background. The
 metric I used 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{total\&space;number\&space;of\&space;blue\&space;pixels}}\times&space;\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{size\&space;of\&space;the\&space;candidate\&space;region}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{total\&space;number\&space;of\&space;blue\&space;pixels}}\times&space;\frac{\mathrm{number\&space;of\&space;blue\&space;pixels}}{\mathrm{size\&space;of\&space;the\&space;candidate\&space;region}}" title="\frac{\mathrm{number\ of\ blue\ pixels}}{\mathrm{total\ number\ of\ blue\ pixels}}\times \frac{\mathrm{number\ of\ blue\ pixels}}{\mathrm{size\ of\ the\ candidate\ region}}" /></a></p>
