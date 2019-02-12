# notes about learning including all concepts 

## list vs. array
python has list and array.
but what is differnce between list and array?
they both can be indexed and iterated through. 
but python array is vector, numpy array cover vector and matrix. 

## list convert to array
```
numpy.array(lists)
```
list has no shape but numpy array has shape concepts.

## construct a batch 
```
data = numpy.concatenate([img_i[np.newaxis] for img_i in imgs], axis=0)
data.shape
```
## about dataset

### how image dimentions are formatted as a single image
```
plt.imread(fname) # return a single image numpy.array.
```

### how images are represented as a collection using a 4-d array
```
data = np.array(imgs) # imgs is list, its elements are np.array, data.shape=[m, H, W, c]
```

### how we can perform dataset normalization
refer to below what is normalization

### what is batch dimensions
many images in an array using a new dimension.
```
np.array(imgs) # get numpy to give us an array of all imgs
```
from this you can get array from (M, N, 3) to (m, M, N, 3), m is the number of img

### what is normalization
we are trying to build a model that undertands invariance. 
we need our model to be able to express all of the things that can possibly change in our data.
this is the first step in understanding what can change. 
it will often start by modeling both the mean and standard deviation of our dataset.
"substracting the mean, and dividing by the standard deviation, another word for this is normalization"

### how to calculate mean and standard deviation
#### mean 
```
mean_img = np.mean(data, axis=0) # data.shape=[100, 218, 178, 3], mean_img is all 100 imgs RGB mean value
                                 # so mean_img.shape=[218, 178, 3]

```
#### std
```
std_img = np.std(data, axis=0)   # simliar as above
```
#### normalization
```
(X - mean) / std  # (x - np.mean(x))/np.std(x)
```
#### but why? why we need to normalize our dataset
after normalization, most of data will be around 0.
if our data does not end up looking like this, then we should either
* get much more data to calculate our mean/std deviation
* either try another method of normalization, such as scalling the values between 0 to 1, or -1 to 1. 
* or possibly not bother with normalization. 
there are other options that one could explore, including different types of normalization such as local contrast normalization for images
or PCA based normalization.

### how we can convert `batch x W x H x channels` 4d array into image original dimension array
```
flattened = data.ravel()
```
for 1 dimension array we can get hist to observe dataset "distribution" or range and frequency of possible values are. 
this is very useful thing to know. it tells us if our data is predictable or not.
```
plt.his(flattened.ravel(), 255)
```

## regarding matplotlib.pyplot operation
1. imread(fname) return imagedata with numpy.array which has shape  
   (M, N) for grayscale images  
   (M, N, 3) for RGB images  
   (M, N, 4) for RGBA images.  

2. imshow(imgdata) imagedata is numpy.array from imread()
so if image is tensor, need to convert to np.array by `tensor.eval(session=sess)`

3. imshow(imgdata)   
imgdata is (H, W, C)
if imagedata shape is (HxW, C)  

```
plt.imshow(imgdata.reshape([H, W, C])) # imgdata is numpy.array, who has reshape() method
```

## regarding numpy
### numpy.array attribute
```
numpy.array.shape
numpy.array.dtype
np.array.astype(np.uint8) # cast
```
### how to create a range of numbers array
```
np.linspace(-3.0, 3.0, 100) # 100 numbers from -3.0 to 3.0
```
### how to convert n-dimension array to 1-dimension array
```
np.squeeze(array)
```

## tf basics
### default graph
```
g = tf.get_default_graph()
```
you can get graph operation by 
```
[op.name for op in g.get_operations()] # if no operation created, this list will be empty `[]`
```
you can get tensor by 
```
g.get_tensor_by_name('LinSpace' + ':0')  # return a tensor
```
you can get tensor shape by
```
tensor.get_shape() or tensor.shape
tensor.get_shape().as_list() # get more friendly format
```
something need to know
```
tensor.get_shape().as_list() # get [100]
```
if you want to get `100`, you need to do 
```
tensor.get_shape().as_list()[0] # get a scaler, a number
```
you can convert tensor to numpy.array by
```
x.eval()
```
you can reshape a tensor by
```
tf.reshape(tensor, shape=[size, size])
```
or using `tf.expand_dims()` to add a singleton dimension at the axis we specify.
below is from [H, W] to [H, W, 1]
```
img_3d = tf.expand_dims(img, 2)
```
or from [H, W, 1] to [1, H, W, 1]
```
img_4d = tf.expand_dims(img, axis=0)
```
what about Variable_scope
```
with tf.Variable_scope("layer/{}".format(i)):
    W = tf.get_variable(name="W", shape=[n, n], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
    b = tf.get_variable(name="b", shape=[n], initializer=tf.constant_initializer(0.0))
```

### tf convNet
* tf convNet api need input image to be 4 dimension, [m, W, H, c]
we have already know how to combine imgs to be a 4d array=> `np.array`
we can use tf reshape by
```
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
```
note: img prelimiarily is 2-d array, [w, h], in order to use tf.convNet, need to convert 2d img to 4d tensor
[1, w, h, 1] or [1, w, h, 3]

### tf shape repression different
for `tf.ones()` like initialize method, params `shape` is (size, size). but for other method like `reshape`, shape is [size, size], `()` and `[]` are the same

### how to understand Gaussian kernel apply to image
convNet actually is image transformation if we have filter or kernel and apply to this image.
filter is functionality used as a sort of detector. 
now from session-1 example, we can see filter changes as per your requirement. 
at first, we have gaussian curve, this is a 1d array, then change this 1d array to 2d filter by multiplying array by its transpose.
then, per tf.conv2d api need kernel to be 4d [h, w, I, o] tensor, so we can use tf.reshape to reshape that 2d array to 4d tensor.
above is for kernel. 
then considering 2d array image, still need to reshape to 4d tensor [1, h, w, c]
after we get two params, we use tf.conv2d() api to calculate. 
this api return 4d tensor, tensor.eval() convert to numpy.array.
finally we can use plt.imshow() to plot result.
imshow can't accept 4d array, we can use tensor[0, :, :, :] or [0, :, : , 0] to saticfy imshow()

### how to understand teache computer to paint image
1. how to undertand one pixel display
```
from skimage.data import astronaut
from scipy.misc import imresize
img = imresize(astronaut(), (64, 64))
plt.imshow(img)
img.shape # (64, 64, 3)

img0 = img[3, 5]
img0.shape # (3, )

plt.imshow(img0.reshape(1, 1, 3))  # you will get this pixel plot
```
2. how to get the whole pic location pixel color infor
```
xs = []
ys = []
for row_i in range(img.shape[0]):
    for col_i in range(img.shape[1]):
        xs.append([row_i, col_i])
        ys.append(img[row_i, col_i])

xs = np.array(xs)
ys = np.array(ys)

xs = (xs - np.mean(xs)) / np.std(xs)
xs.shape, ys.shape
```

### tf.placeholder
```
img = tf.placeholder(tf.float32, shape=[None, None], name='img')
```
we can create placeholder for our parameters, if shape is not specified, we can feed a tensor of any shape. .
```
mean = tf.placeholder(tf.float32, name='mean')
sigma = tf.placeholder(tf.float32, name='sigma')
ksize = tf.placeholder(tf.float32, name='ksize')
```

### mini Batch
#### how to create mini-batch index
```
idxs = np.arange(100)
batch_size = 10
n_batches = len(idxs) / batch_size
for batch_i in range(n_batches):
    print(idexs[batch_i * batch_size : (batch_i + 1)*batch_size])
```

random idx
```
rand_idxs = np.random.permutation(idxs)
```
## regarding Gaussian curve
* this curve is 1-dimension array data performace given mean and sigma(std devicative)
* we can create a caussian filter for convNet calculation.

### how to creat a 2-d Gaussian. 
this can be done by multiplying a vector by its transpose. 
so we can get this matrix by `tf.matmul(z, tf.transpose(z))`
```
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
```
### how to convert 2d Gaussian to 4d format[H, W, I, O]
```
z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])
```

# Questions
* how to convert [H, W, 3] to [H, W, 1]

```
img = plt.imread(fname)             # this return [H, W, 3]
imgx = plt.imread(fname)[..., 2]    # => [H, W]
imgx = tf.reshape(imgx, [H, W, 1])  # => 
```

# zipfile usage
```
import zipfile, os
zipf = zipfile.ZipFile(filename, 'w', zipfile.DEFLATED)
if (os.path.exists(file)):
    zipf.write(file)
zipf.close()
```
# basic concepts
* gradient descent: is a way to optimize a set of parameters.
* learning rate: how far along the gradient we should travel, typically set this value from anywhere between 0.01 to 0.00001
* actually all the possibility can be trained.
* Unsupervised vs.supervised learning
  * unsupervised learning  
    - you have a lot of data and you want the computer to reason about it, maybe encode the data using less data.
just exploe what patterns these might be.this is useful for clustering data, reducing the dimensionality of the data,
or even for generating new data.
  * supervised learning
    - you actually know what you want out of your data.
* Autoencoders, a type of nn that learns to encode its inputs, often using much less data.it tries to output whatever it was given as input.

# pyplot basic operation
## example1
```
fig = plt.figure()
ax = fig.gca()
x = np.linspace(-3.0, 3.0, 100)
y = x\*\*2
ax.plot(x, y)
ax.set_ylabel('Cost')
ax.set_xlabel('something')
plt.show()
```
## 3d example
```
from mpl_toolkits.mplot3d import axes3d
import matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection="3d")
x, y = np.mgrid[-1:1:0.02, -1:1:0.02]
X, Y, Z = x, y, x+y
ax.plot_surface(X, Y, Z, rstride=2, cstride=2, alpha=0.75, cmap='jet')
```
## plt normal usage
```
n = tf.random_normal([1000], stddev=0.1).eval(sessoin=sess)
plt.hist(n)     # plt.hist() param is numpy.array
plt.show()
```
## how to add more curve into one plot
```
x = ...
y = ...
z = ...
fig, ax = plt.subplots(1, 1)
ax.scatter(x, y)
ax.plot(x, z)
plt.show()
```

another example

```
x = np.linspace(-6, 6, 1000)
plt.plot(x, tf.nn.tanh(x).eval(session=sess), label="tanh")
plt.plot(x, tf.nn.sigmoid(x).eval(session=sess), label="sigmoid")
plt.plot(x, tf.nn.relu(x).eval(session=sess), label="relu")
plt.legend(loc="lower right")  # display legend label
plt.xlim([-6, 6])
plt.ylim([-2, 2])
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid('on')
plt.show()
```

#### multiple line in one plot
```
t = np.arange(0., 5., 0.2)
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
```
#### categorical 
```
names = ['a', 'b', 'c']
values = [1, 10, 100]

plt.figure(1, figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)

plt.subplot(132)
plt.scatter(names, values)

plt.subplot(133)
plt.plot(names, values) # default is line plot
plt.suptitle('name and value')
plt.show()
```

### why these three curve works for learning
this curve actually regulate output Y_pred value. so that learning operation works.
so sigmoid always output -1, 1 output, while relu output 0 and 1, can be work as the yes or no classification.
tanh works better than sigmoid, but its much calculation make sigmoid used normally.

### pkmital/CADL
don't worry if a lot of it doesn't make sense, and it really takes a bit of practice before it starts to come together.

## tips

### how to create gif
```
from libs import utils, gif
gif.build_gif(imgs, saveto="xx.gif") # imgs is list or np.array
```
### how to save list-like pic to lists of pics
```
imgs = [np.clip(m, 0, 255).astype(np.uint8) for m in img_lists]
imgs_data = np.array(imgs)
imgs_data.shape
for m_i in range(imgs_data.shape[0]):
    plt.imsave("finals/f_" + str(m_i), imgs_data[m_i, :, :, :])
```
### how to get files list from directory
```
files = [os.path.join("foldname", file_i) for file_i in os.listdir("flodname")] 
```

### how to get imgs list from files list
```
imgs = [plt.imread(f_i) for f_i in files]
```

## CADL libs utils
#### example1
```
from libs.utils import montage
imgs = ds.X[:1000].reshape((-1, 28, 28))
plt.imshow(montage(imgs), cmap='gray')
```

#### how to understand image input expression
1. [m, HxW]  => tf 
2. [m, H, W] => imshow()
3. both tf and np have reshape() method

#### how to concat string with different format data
```
# {}.format(value) => value can be integer, string, floating
print("{} traceback".format("integer"))
print("Hello, I am {} year old".format(23))
```

#### how to understand `enumerate()`
enumerate is useful for obtaining an indexed list: (0, seq[0]), (1, seq[1]), (2, seq[2]), (3, seq[3])

#### python list operations
1. list[::-1] change from [1, 2, 3, 4] to [4, 3, 2, 12]
2. list[::2] change from [1, 2, 3, 4] to [1, 3] => even index
3. list appends: list.append(value) or list = list + [value]

#### how to load pretrained network
```
from libs import inception
net = inception.get_inception_model()
net['labels'] # => get labels list
tf.import_graph_def(net['graph_def'], name='inception')
g = tf.get_default_graph()
names = [op.name for op in g.get_operations()]
\# print(names) => to see all the operations in loaded model
input_name = names[0] + ':0'
x = g.get_tensor_by_name(input_name)
softmax = g.get_tensor_by_name(names[-1]+':0')
res = np.squeeze(softmax.eval(feed_dict={x: img_4d}))

res = np.mean(res) # or res = np.max(res)
res = res / np.sum(res)

infers = [(res[idx], net['labels'][idx]) for idx in res.argsort()[-5:][::-1]]

```

#### how to visualizing pretrained model's filters
why we visualize filters? in order to know what filters do.
why we visualize features(actually neurons)? in order to know filtered image looks like.

why we visualize gradient? to use backprop to show us the gradients of a particular neron
with respect to our input image to undertand what the deep layers are really doing. 
but gradients will look like what? 
let's visualize the network gradient activation when backpropagated to the original input image.
this is effectively telling us which pixel are responding to the predicted classes or given neuron. 
author says, we use a forward pass up to the layer that we are interested in, and then a backprop to help us understand what pixels in particular contributed to the final activation of that layer. we will need to create an operation which will find the max neuron of all activations in a layer, and then calculate the gradient of that objective with respect to the input image. 
[cmt]basically I understood. for pretrained model, every layers parameters are frozen, when you push input into any layer, you will get the same effection when training. for each layer, every pixel contribute. 
so author says, we will find the max neurons of activation in that layer, and to see what pixel works for by using gradient of max value with respect to image. actually for this max value, all pixels of image are contribute. 
but what this will look like?

#### how to understand the process of learning a new language
1. A new language is actually a new expression which that new interpretor can understand. 
2. every language basically has its own power toolkits, this is the reason why it get people involved, except C language. but C is flexiable and extendable, just need developer take care of everything they will use. but for other language, they have tools, just find it and use the best of them.
3. basic rule of learning a new language is to use it to implement what you want. dont' hate making mistakes.  
4. practice actually is implementation.

#### how to undertand style
1. load model
2. get content image and style image
3. find a deeper layer used to calculate content feature, `g.get_tensor_by_name(layer_name).eval(feed_dict={x:img})` to get content feature. =>from analysis, this feature is Variable.
4. find all conv layers used to calculate style feature, still use feature method to get a style activations list per each conv layer. which is the raw data, need to convet to gram_matrix. matmul(x.T, x).
5. make content_loss operation, this loss can be trained. `tf.nn.l2_loss((current_feature - content_features)/content_features.size)` => need input image as placeholder. 
6. make style_loss operation, `tf.nn.l2_loss((current_layer_gram_feature - style_feature)/style_feature.size)`
7. make the total_loss, `alpha * content_loss + beta * style_loss`, and optimizer
7. train


# todo
1. how to debug python
2. how to save list, array or something else, and load them when needed.
3. how to split the whole process into sub steps, and every steps are indepent.
4. make sess4.py work using placeholder. currently it use interactiveSession works, but how to convert to normal process, you need enfort to work on it. [this is the learning rules, get chanlge and resolve it, you will improve. this process is not easy, but full of excitement. in contrast, leave or run away when chanle happens, you will always walk around the shallow water, never can be able to dive into the essentials.]

