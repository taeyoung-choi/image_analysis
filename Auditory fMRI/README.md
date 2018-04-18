### Assignment 2

#### 1. Realignment

It is often the case that a person's head does not stay still during the acquisition of fMRI data. Even a small movement can largely affect the quality of data. Therefore, we want to correct movements of the head by aligning the data to a reference image.

SPM12 calculates six parameters for translation and rotation for image realignment. The first three parameters are for translation, which contains relative (x, y, z) coordinates of the image. The latter three parameters are for rotation that can be used after we re-center the image by the translation. Rotation parameters, which are pitch, roll and yaw, will rotate the image in the three dimensinal space to align it to the reference image. We are using 84 images, and we have 6 parameters for each image.

#### 2. Coregistration

We now have the brain images that removed movements during the scan. Therefore, we can capture functional activations of the brain more accurately. We would like to overlay functional activations over the subject's own anatomy. This is called coregistration. A structural image has high resolution and provides more details about the subject's anatomy. However, fMRI images are blurry and contain image distortions. So coregistration allows us to see the functional activations more clearly on the structural image. The reference image is the average of aligned images we get from Step 1.

Coregistration calculate a spatial filter that maximizes the mutual information between the reference image and the structual image. 

#### 3. Segmentation

A brain has different types of tissues, which are grey matter, white matter, cerebralspinalfluid (CSF), skull and soft tissue. Each type is resposible for different functions in the brain.

The first process of segmentation is a bias correction. fMRI images may contain bias artifact. Having changes in intensities due to the poor quality of images can lead to a misclassification of tissue classes. Thus, it is essential to correct bias artifact and acheive uniform intensities within the different types of tissues before we segement them.

Then we use the pre-defined tissue probability map as priors and update the priors based on the subject's brain images. The classification model can deform the tissue probability map using the parameters we defined in previous steps. The model employs mixture of Gaussians, because a voxel contains signal from a multiple tissue types. Gaussian mixture models are more robust to the parts that fall betweeen the intensities of the pure classes. The deformation of the tissue probability map is called warping. SPM 12 uses non-linear warps by default.

As a result, SPM 12 creates grey and white matter images, a bias corrected structural image and a deformation field.

#### 4. Normalization

In order to make different studies comparable, we must report the results in the standard coordinate system. The structure of a brain varies by each subject. In the previous step, we deformed the tissue probability map to fit it to the actual brain images. We need to apply a inverse transformation of the deformation field to the subject's images to get the normalized images.

#### 5. Smoothing

This step smoothes image volumes with a Gaussian kernel. Smoothing can suppress noise and effects that created from the functional and anatomical differences during inter-subject averaging. Smoothing will also make data more normally distributed. We can vary the width of a Gaussian kernel if we are interested in different parts of the brain.

#### 6. Specify

In order to perform a statistical analysis of fMRI data using General Linear Models, we need to specify the GLM design matrix, fMRI data files and filtering. One of the required parameters that is needed to construct the desing matrix is timing parameters. We set TR to 7, which is the interscan interval that this data was collected. The data should have the same image dimensions, orientations and voxel size. Therefore, we use the smoothed and normalized images that we have generated in the previous steps. Onsets specifies the begining of a event and Duration specifies the event duration. After setting the specification of the model, SPM 12 will create a SPM.mat file containing the specified design matrix.

#### 7. Estimate

This step evaluates the General Linear Model with the specification we defined in step 6.


#### 8. Limitations
- Modeling Categorical Responses: We might have different effect based on the different bi-syllabic words. We might want to test categorical responses for different words. 

#### 9. Interpretation

There are 13 clusters that are significant. The coordinates in MNI space for each maximum are [57, -22, 11], [-63, -28, 11], [30, -31, -16], [54, 2, 47], [-63, -55, -7], [-30, -31, -19], [-48, 44, 5], [48, 26, 23], [45, 17, 23], [-57, -1, 44], [66, -43, -16], [57, -43, 53] and [-54, -4, 47]. These are the significant clusters as a result of T-test, where the regression coefficients are greater than zero. So for a given auditory stimulation, these parts of the brain are responsible.


