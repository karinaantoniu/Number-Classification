<!DOCTYPE html>
<html>
<body>

<h1> <strong> MNIST Digit Classification with Convolutional Neural Network </strong> </h1>
<p>
	One of the applications in computer vision is MNIST Digit Classification, and I used multiple methods to achieve the best accuracy on the dataset while optimizing for minimal resource consumption.
</p>

<h1> <strong> Sequential CNN Architecture </strong> </h1>
<p>
	The CNN architecture follows a progressive feature extraction approach: <br>
	<br>
	<i> First Convolutional Block </i> utilizes 32 filters with a 3×3 kernel and same padding to extract low-level features such as edges and corners. The MaxPooling layer creates spatial hierarchy while Dropout (0.3) provides initial regularization against overfitting.<br>
	<br>
	<i> Second Convolutional Block </i> deepens the network with 64 filters for medium-level feature extraction. BatchNormalization stabilizes training and accelerates convergence, followed by ReLU activation for non-linear transformation. Dropout rate increases to 0.4 to handle the growing complexity.<br>
	<br>
	<i> Third Convolutional Block </i> employs 128 filters for high-level feature detection. MaxPooling with stride 2 reduces dimensionality efficiently, while Dropout (0.5) provides strong regularization before the classification stage.<br>
	<br>
	The architecture concludes with a Flatten layer converting 3D feature maps to 1D vectors, feeding into a Dense layer with softmax activation for 10-class classification.<br>
	<br>
	With this progressive filter increase for a hierarchialfeature learning I obtainted the following results: <br><br>
	<img src="Screenshot 2025-09-27 at 08.29.08.png" width="400" height="100" style="display: block; margin: auto;">
</p>

<h1> <strong> Depthwise Separable Convolution </strong> </h1>
<p>
	Unlike the standard method (the previous one) that perform both spatial filtering and channel combination simultaneously, Separable Convolution splits the procces into 2 different parts: the depthwise convolution and the pointwise convolution. This method provides better accuracy while maintaining the same resource consumption. The depthwise conv applies a single filter on each input channel capturing the details independently while the pointwise convolution combines this features. This decomposition reduces the parameters count making the model more efficient and less prone to overfitting.<br>
	<br>
	<img src="Screenshot 2025-09-27 at 18.51.35.png" width="400" height="100" style="display:block; margin: auto;">
</p>

<h1> <strong> Mobile Net V2</strong> </h1>
<p>
	Using MobileNetV2 for the MNIST dataset is an excessive approach that leads to suboptimal accuracy for several reasons. MobileNetV2 has a lot of parameters, making it overly complex for classifying simple 28x28 pixel handwritten digits. Such a deep architecture is unnecessary given the simplicity of MNIST and often results in overfitting. This observation can be seen in the accuracy score obtainted. <br>
	<br>
	<img src="Screenshot 2025-09-27 at 19.46.10.png" width="400" height="100" style="display:block; margin:auto;">
</p>

</html>
</body>