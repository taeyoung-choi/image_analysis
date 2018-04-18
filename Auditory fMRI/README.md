### Assignment 2

#### 1. Realignment

It is often the case that a person's head does not stay still during the acquisition of fMRI data. Even a small movement can largely affect the quality of data. Therefore, we want to correct movements of the head by aligning the data to a reference image.

SPM12 calculates six parameters for translation and rotation for image realignment. The first three parameters are for translation, which contains relative (x, y, z) coordinates of the image. The latter three parameters are for rotation that can be used after we re-center the image by the translation. Rotation parameters, which are pitch, roll and yaw, will rotate the image in the three dimensinal space to align it to the reference image. We are using 84 images, and we have 6 parameters for each image.

#### 2. Coregistration

We now have the brain images that removed movements during the scan. Therefore, we can capture functional activations of the brain more accurately. We would like to overlay functional activations over the subject's own anatomy. This is called coregistration. A structural image has high resolution and provides more details about the subject's anatomy. However, fMRI images are blurry and contain image distortions. So coregistration allows us to see the functional activations more clearly on the structural image. The reference image is the average of aligned images we get from Step 1.

Coregistration calculate a spatial filter that maximizes the mutual information between the reference image and the structual image. 

#### 3. Segmentation

A brain has different types of tissues, which are grey matter, white matter, cerebralspinalfluid (CSF), skull and soft tissue. Each type is resposible for different functions in the brain.

The first process of segmentation is bias correction. fMRI images are not
