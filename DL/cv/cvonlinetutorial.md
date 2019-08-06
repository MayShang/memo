## how

## what 
### imgProc

#### Smoothing
C++:
  - different blur functions calling
  - WaitKey(delay) delay usage
  - different kernel yield different results

### Eroding and Dilating
this is called morphological operations, meansing, process image based on shapes.
applying a structuring element to an input image and generate an output image.

1. why  
in order to:
- removing noise
- isolation of individual elements and joining disparate elements in an image.
- finding of intensity bumps or holes in an image.

2. referrence code
ImgProc/Morphology_1.cpp
ImgProc/Morphology_1.py

3. tips
  - C++ 
    - CommnadLineParser usage
    - Mat src.empty() to tell if image open or not

  - Python
    - Mat `src is None`

### more morphology transformations
`cv::morphologyEx`
1. usage
 * Opening
 * Closing
 * Morphological Gradient
 * Top Hat
 * Black Hat

2. theory
 - Opening
   * dst = open(src, element) = dilate(erode(src, element))
   * useful to remove small objects

 - Closing
   * dst = close(src, element) = erode(dilate(src, element))
   * useful to remove small holes

 - morphological Gradient
   * dst = morph(src, element) = dilate(src, element) - erode(src, element)
   * useful to find the outline of an object

 - Top Hat
   * dst = tophat(src, element) = src - open(src, element)
 - Black Hat
   * dst = blackhat(src, element) = close(src, element) - src

3. reference code
`ImgProc/Morphology_2.cpp`

### line detection
1. reference code
`imgProc/morph_lines_detection.cpp`

2. application dilate and erosion
 - there is threshold application
 - extract something out app

3. tips
what is binary image?
white fg and black backgroud.

4. steps:
 - gray
 - inverse and application of adaptiveThreshold gray image to be binary image
 - apply horizontal kernel to extract horizontal image (interesting!!)
 - apply vertical kernel to extract vertical image.
 - smoothing vertical image `vertical[rows, cols] = smooth[rows, cols]`

### threshing
1. theory
  - The simplest segmentation method
  - Application example: Separate out regions of an image corresponding to objects which we want to analyze. This separation is based on the variation of intensity between the object pixels and the background pixels.
  - To differentiate the pixels we are interested in from the rest (which will eventually be rejected), we perform a comparison of each pixel intensity value with respect to a threshold (determined according to the problem to solve).
  - Once we have separated properly the important pixels, we can set them with a determined value to identify them (i.e. we can assign them a value of 0 (black), 255 (white) or any value that suits your needs).

2. reference code
`imgProc/threshold.cpp`

3. tips
think when to use different threshing type?

### threshold inRange
1. theory
  - HSV colorspace, H:hue, S:saturation, V:value. similar to RGB color model. hue channel models the color type. it is very useful in image processing tasks that need to segment objects based on its color. variation of saturation goesfrom unsaturated(shade of gray) and fully saturated. value channel describes the intensity of the color. refer to wikimeida HSV cylinder.

2. refernce code
`imgProc/threshold_inRange.cpp`

### making your own linear filter
1. theory
  - `correlation`: corralation is an operation between every part of an image and an operator(kernel).
  - `kernel`: a kernel is essentially a fixed size array of numerical coefficients along with an anchor point in that array, which is typically located at the center.
  - how they works? correlation and kernel.
    * Assume you want to know the resulting value of a particular location in the image. The value of the correlation is calculated in the following way:
      * Place the kernel anchor on top of a determined pixel, with the rest of the kernel overlaying the corresponding local pixels in the image.
      * Multiply the kernel coefficients by the corresponding image pixel values and sum the result.
      * Place the result to the location of the anchor in the input image.
      * Repeat the process for all pixels by scanning the kernel over the entire image.

2. reference code
`imgProc/filter2D_demo.cpp`

### adding borders to your images
1. use `copyMakeBorder()`

2. theory
  - `BORDER_CONSTANT`: pad the image with a constant value (black or 255)
  - `BORDER_REPLICATE`: the row or column at the very edge of the image is replicated to the extra border.

3. reference code
`imgTrans/copy_make_border.cpp`

4. tips
C++:
  - Mat.rows() get rows number, Mat.Size() get rows X cols

### sobel derivatives
1. theory
  - sobel operator is a discrete differentiation operator. it computes an approximation of the gradient of an image intensity function.
  - the sobel operator combines Gaussian smoothing and differentiation.
  - generates output image with the detected edges bright on a darker background.

### Laplace operator
1. theory
  - use the second derivative (==0) to detect edges in an image. regarding zero in other meaningless locations, can be solved by applying filtering where needed. (but how, how to tell it's not edges).
  - uses the gradient of image, it calls internally the sobel operator to perform its computation.

2. reference code
`ImgTrans/Laplacian_Demo`

3. tips
steps:
 - loads image
 - remove noise by applying a Gaussian blur and then convert the original image to grayscale
 - apply a Laplacian operator

### Canny Edge Detector
1. theory
  - optimal detector
  - low error rate (only existent edges), good localization (the distance between detected edge pixels and real edge pixels to be minimized.) only one detector response per edge.

### Hough Line Transform
1. theory
  - it is a transform used to detect straight lines.
  - an edge detection pre-processing is desirable, in order to apply the transform.
  - line in image can be expressed: two coordinate syste a.Cartesian and Polar coordinate system.
  - Hough transforms, express line in the Polar system.
  - if the curves of two different points intersect in the plane (theta - r), that means that both points belong to a same line.
    this means a line can be detected by finding the number of intersections between curves. the more curves intersecting means that the line represented by that intersection have more points.
    in general, we can define a threshold of the minimum number of intersections needed to detect a line.
  - this is what the Hough Line trans does. it keeps track of the intersection between curves of every point in the image. if the number of intersections is above some threshold, the it declares it as a line with the params (theta, r) of the intersection point.

### Histogram Equalization
1. theory
  - intensity distribution of an image to another distribution. to another distribution.
  - each color channel
  - improve the constrast in an image, in order to stretch out the intensity range.
  - how to equalizate? mapping one distribution (the given histogram) to another distribution.
