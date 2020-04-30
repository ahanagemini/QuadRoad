# Road Segmentation using Multi-modal Data and MUlti-class Ground truth
Project for using aerial images, Lidar data, multi-spectral images to perform segemntation of paved roads by using binary and multi-class ground truth. Four different models are trained and then the softmax values for each pixel are combined used weighted averaging. The weights are computed using differential evolution to optimize the global intersection over union by using th evalidation data.  
The paper is under review for KES conference. <br/>

<b>Project Information: </b> <br />
The project performs the following steps:  <br />
1. For the aerial images, Lidar images (1 channel) with binary ground truth and aerial images with multi-class ground truth, the following models are available: <br />
	1. SegNet <br />
		https://arxiv.org/abs/1511.00561 <br />
	2. SegNet with Atrous convolution se the code for src/model/model\_A.py for details (SegNet\_A) <br />
	3. SegNet with atrous convolution and leaky relu (SegNet\_AL) <br />
	4. DeepLabv3+ not a part of the pytorch code but under deeplab\_code <br />
		Code: https://github.com/tensorflow/models/tree/master/research/deeplab <br />
		Paper: https://arxiv.org/abs/1802.02611 <br />
2. For the multi-spectral images, resolution is 192*192 instead of 500*500 as in case of the rest. <br />
	1. SegNet\_hs, which can handle the resolution disparity in case of multi-spectral images and takes 8 channels as input. see the code in src/models/model\_hs.py for details. <br />
	2. SegNet\_hs\_A uses atrous convolutions in multi-spectral model <br />
	3. SegNet\_hs\_AL uses atrous convolutions and leaky relu. <br />
	4. Segnet\_hs\_ALD same as SegNet\_hs\_AL but uses a dropout. <br />
3. For augmenting the images, we use intensity shift for each channel and random roatation by a multiple of 90 degree<br />
                Codes are in src/utils. create-augmented_images.py, create_rotated_images.py and create_rotated_hs.py (for multi-spectral images only) <br />
4. Differential evolution is applied on the 4 predictions for the validation set to find weights and a threshold. If the weighted average is greater than threshold, the pixel is classified as a a road pixel <br />
	Code is from scipy.optimize differential_evolution <br />
	Differential evolution was proposed by Storn and Price https://www.researchgate.net/publication/227242104_Differential_Evolution_-_A_Simple_and_Efficient_Heuristic_for_Global_Optimization_over_Continuous_Spaces. <br />
5. Final segmentation can be generated for the test set using the computed weights or using equal weights by providing the weights as the inputs. Segmentation genrated can be optionally saved. <br />

<b>Requirements: </b> <br />
1. For all the code under src: install packages in requirements.txt <br />
2. For deeplab code: install a separate conda environment with requirements\_deeplab.txt <br />

