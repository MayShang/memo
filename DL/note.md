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

another explaination:
it's simply a case of getting all data on the same scale: if the scales for different features are widely different, this can have a knock-on effect on you ability to learn.

and from wiki: feature scaling
standardize the range of independent variables or features of data. in data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

gradient descent converges much faster with feature scaling than without it.
* rescaling (min-max normalization), the simplest method and rescaling the range to [0, 1] or [-1, 1], x' = (x - min(x)) / (max(x) - min(x))
* mean normalization, x' = (x - averaage(x)) / (max(x) - min(x))
* standardization, x' = (x - mean(x)) / std(x)
standard deviatrion, sqrt(sum(square(x)/n))
* scaling to unit length, x' = x / ||x||, to scale the components of a feature vector such that the complete vector has length one. this usually means dividing each component by the Euclidean length of the vector. 

* variance, the average of the squared differences from the Mean.
`www.mathsisfun.com/data/standard-deviation.html`
standard Deviation, distances from the mean, sqrt(sum(square(x-mean(x)))/n), [mean-sd, mean+sd]
mean squared error, distances from the real value, sqrt(sum(square(x-x'))/n)

* `tf.random_unifrom`, uniform distribution, is a distribution that has constant probability.
* `tf.random_normal`, bell curve

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

* tensor, tensor is a symbolic handle, this handle is one of the output of an operation. it does not hold the values of that operation. but instead provides a means of computing those values in session. So tensor is not variable, it is JUST a handle, it provides computation. 
* t.eval() is a shortcut for calling `tf.get_default_session().run(t)`
so in `sess.run(op, feed_dict={})` =>`op.eval(feed_dict={})`

* GAN, generative adversarial network. it is actually two networks. one called the generator, and another called the discriminator. The basic idea is the generator is trying to create tlhings which look like the training data. so for images, more images that look like the training data. the discriminator has to guess whether what its given is a real training example. or whether its output of the generator. 
by training one after another, you ensure neither are ever too strong, but both grow stronger together. the discriminator is also learning a distance function! this is pretty cool because we no longer need to measure pixel-based distance, but we learn the distance function entirely!
if we finally get this distance function, we can store this function and all params, for a new image, we don't need to compute distance, instead, we get code from generator, and then plug into the distance function, we get what we expect. so pretty cool!

* recurrent neural network, let us reason about information over multiple timesteps.
they are able to encode what it has seen it the past as if it has a memory of its own. 
it doesn this by basically creating one HUGE network that expands over time. 
it can reason about the current timestep by conditioning on what it has already seen.
by giving it many sequences as batches, it can learn a distribution over sequences which can
model the current timestep given the previous timesteps.

1. a huge network
2. remember past timesteps 
3. reason about the current timestep.

* VSM, vector space model. two distribution method: count-based method and predictive model.
for predictive mode, there are two implementations, CBOW, predicts target word from context(predict 'mat', from 'a cat sit on the '); skip-gram, predicts context words from the target words.
for most part, it turns out skip-gram is a useful method for larger dataset.

### how to undertand default session and default graph
```
graph1 = tf.Graph()
graph2 = tf.Graph()

with graph1.as_default() as graph:
  a = tf.constant(0, name='a')
  graph1_init_op = tf.global_variables_initializer()

with graph2.as_default() as graph:
  a = tf.constant(1, name='a')
  graph2_init_op = tf.global_variables_initializer()

sess1 = tf.Session(graph=graph1)
sess2 = tf.Session(graph=graph2)
sess1.run(graph1_init_op)
sess2.run(graph2_init_op)

\# Both tensor names are a!
print(sess1.run(graph1.get_tensor_by_name('a:0'))) # prints 0
print(sess2.run(graph2.get_tensor_by_name('a:0'))) # prints 1

with sess1.as_default() as sess:
  print(sess.run(sess.graph.get_tensor_by_name('a:0'))) # prints 0

with sess2.as_default() as sess:
  print(sess.run(sess.graph.get_tensor_by_name('a:0'))) # prints 1

with graph2.as_default() as g:
  with sess1.as_default() as sess:
    print(tf.get_default_graph() == graph2) # prints True
    print(tf.get_default_session() == sess1) # prints True

    \# This is the interesting line
    print(sess.run(sess.graph.get_tensor_by_name('a:0'))) # prints 0
    print(sess.run(g.get_tensor_by_name('a:0'))) # fails

print(tf.get_default_graph() == graph2) # prints False
print(tf.get_default_session() == sess1) # prints False
```
you can't run an operation in graph2 with sess1. this example is typical.
so what the rules to assign session and graph in training learning. 

### what are the rules of design session and graph


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

#### how to save plot
```
plt.savefig("xx.png")
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

#### python zip
zip: tuple
```
vocab = list(set(txt)) # len(vocab) is all character number in txt, len(txt) is text file length.
encoder = dict(zip(vocab, range(len(vocab)))) # 'v':76 return tuple pair
decoder = dict(zip(range(len(vocab)), vocab))

```

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
the points for loaded pretrained model:
1. the placehoder of this model is the first layer tensor
```
x = g.get_tensor_by_name(input_name)
```

2. for other layers operation, `x` is always the placeholder tensor to run.
```
softmax = g.get_tensor_by_name(names[-1]+':0')        # operation
res = np.squeeze(softmax.eval(feed_dict={x: img_4d})) # equal to `sess.run(softmax, feed_dict={x:img_4d})`
```
3. for this 'x' still a placeholder which need to feed_dict during 'run', so some value still keep unknow untill after `sess.run()`

```
def get_gram_matrix(g, name):
    layer_i = g.get_tensor_by_name(name)
    layer_shape = layer_i.get_shape().as_list()

    print(layer_shape)     ======> shape will be [?, ?, ?, 512]

    layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3] ===> will be wrong if input is `x`

    layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
    # gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
    gram_matrix = tf.transpose(layer_i)
    print(gram_matrix.shape)
    return gram_matrix

x = g.get_tensor_by_name(names[0] + ':0')

layer_name = 'vgg/conv4_2/conv4_2:0'
gram_op = get_gram_matrix(g, layer_name)

with tf.Session(graph=g) as sess:
    s = sess.run(gram_op, feed_dict={x:img_4d})
    print(s.shape)  =====> this working, because it's valid after running

```

4. we can use `input_map` to map input and get everything inside the network prefixed before run

```
g = tf.Graph()
net = vgg16.get_vgg_model()

with tf.Session(graph=g) as sess, g.device('/cpu:0'):
    # net_input = tf.Variable(img_4d)
    net_input = tf.get_variable(name='input', shape=(1, 224, 224, 3), dtype=tf.float32,
                               initializer= tf.random_normal_initializer(mean=0.0, stddev=0.5))
    
    tf.import_graph_def(net['graph_def'], name='vgg', input_map={'images:0':net_input})

    layer_name = 'vgg/conv4_2/conv4_2:0'
    gram_op = get_gram_matrix(g, layer_name)

    sess.run(tf.global_variables_initializer())

    s = sess.run(gram_op)
    print(s.shape)

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

#### pretrained model resources
```
http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models
```

### save and restore models
#### restore
```
sess = tf.Session()
\#load meta graph and restore ckpt
saver = tf.train.import_meta_graph('./../xx.meta')
saver.restore(sess, tf.train.latest_checkpoint('./../'))
print(sess.run('b1:0')) # Variables can be access directly
g = tf.get_default_graph()
w1 = g.get_tensor_by_name('w1:0') # placeholder
feed_dict = {w1: 13}
op_to_restore = g.get_tensor_by_name('op_to_restore:0')
print(sess.run(op_to_restore, feed_dict))
```
#### inspect variables in a checkpoint
```
\#import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1. ]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1. ]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
```

### how to save and restore models
use `SaveModel`to save and load your model, model includes variables, graph, and the graph's metadata.
the saveModel will load in tensorflow serving and support the predict API. to use the classify, regress, or multiInference APIs.
SaveMode apis to support above.
if you need more custom behavior, you'll need to use the SaveModelBuilder.

#### features in SaveModel
1. multiple graphs sharing a single set of variables and assets can be added to a single SaveModel. Each graph is associated with a specific set of tags to identification during a load or restore operation.
2. support for SignatureDefs
  * Graph that are used for inference tasks typically have a set of inputs and outputs. this is called a `Signature`.
  * SaveModel uses `SignatureDefs` to allow generic support for signatures that may need to be saved with the graphs.
  * how to use SignatureDef？
3. support for Assets
  * for cases where ops depend on external files for initialization, such as vocabularies, SaveModel supports this via assets.
  * assets are copied to the SaveModel location and can be read when loading a specific meta graph def.
4. support to clear devices before generating the SaveModel.

#### SaveModel
SaveModel manages and builds upon existing Tf primitives such as Tf Saver and MetaDef. The Saver is primarily used to generate the variable checkpoint.
#### SaveModel Components
1. SaveModel protocol buffer: `saved_model.pb` or `saved_model.pbtxt`, the graph definitions as `MetaGraphDef` protocol buffers.
2. Assets
3. extra assets
4. Variables, subfolder called 'variables' and Saver output `variables.data-??-of-???`



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

## word2vec
### reference
search `mccormickml` and 'applying word2vec'


### how to understand word2vec batch
this batch generator uses gram-skip method. meaning, predict context from target words.
1. we have a data pool, every epoch we have a bunch of batch data to train. 
2. for each batch, we have a skip window, skip window is left and right word of target word. skip window has a span = 2 * skip_windwo +1 => [skip_window target skip_window]
3. num_skips means how many times each word will be used. 
4. for batch, we have batch/num_skips pairs of target and context. list batch[] correspond to target word, labels[] to context word.
```
data = [195, 2, 3134, 46, 56]

batch = [2,   2,    3134, 3134, 46,    46, 56, 56]
labels= [195, 3134, 46,   2,    3134,  56, 46, x ]
```
implementation
```
    data_index = 4
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
      context_words = [w for w in range(span) if w != skip_window]
      words_to_use = random.sample(context_words, num_skips)
      for j, context_word in enumerate(words_to_use):
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[context_word]
      if data_index == len(data):
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
```
### mccormickml note
1. we are going to train a simple nn with a single hidden layer to perform a certain task, but then we are not actually going to use that nn for the task we trained it on! instead, the goal is actually just to learn the weights of the hidden layer - then these weights are actually the 'word vectors'.
so now embeddings are the params we're trying to learn. we use these word vectors. 

2. given a specific word in the middle of a sentence(the input word), look at the words nearby and pick one at random. the network is going to tell us the probability for every word in our vocabulary of being the 'nearby word' the we chose. ( comm: we get one word in the middle of a sentence, predict the nearby words. send this input word to network, it will give us the probability of all vocabulary to be this input word 'nearby word' )
3. how to train this network. we train this network by feeding it word pairs found in our training documents. when training finished, if you give the word 'xx' as input, then it will output a much higher probability for some pair than rest pairs. 
4. we are going to learn word vectors with 300 features. 
so the hidden layer weight matrix is [10000, 300], input one word vector [1, 10000]
so [1, 10000] \* [10000, 300] = [1, 300] 
and the end goal of all of this is really just to learn this hidden layer weight matrix.
5. the [1, 300] word vector is one word features vector, then fed to the output layer. which is a softmax regression classifier. it will produce an oputput between 0 and 1, and the sum of all these output values will add up to 1. 
6. about the output layer. word vector will be sent to output layer, and more over, each output neuron has a weight vector [300, 1] feature, weight vector. this weight vector is learned.
so in summary, this network is simply but large. one hidden layer, we need this hidden layer weights matix. and what about the output neuron weight vector? we feed one hot encoded word, calculate weight to yeild feature vector, then this vector go to 10000 neuron's softmax to get probability over 10000. 

7. what is cosine similarity. it tends to be useful when trying to determine how similar two texts/documents are. Cosine similarity works, because we ignore magnitude and focus solely on orientation.o
let's say we have 2 vectors, each representing a sentence. if the two vectors are close to parallel, maybe we assume that both sentences are 'similar' in theme. 
np.dot(A, B) similar to cosine theta, if cosine theta = 1, means A and B are the same orientation, and they are similar.

## how to understand np and tf random
### random seed
`np.random.seed(0)` makes the random numbers predictable
```
np.random.seed(0)
np.random.rand(4) # => array([0.55, 0.72, 0.6, 0.54])
np.random.seed(0)
np.random.rand(4) # => array([0.55, 0.72, 0.6, 0.54])
```
with the seed reset every time, the same set of numbers will appear every time.
if not, different numbers appear with every invocation:
```
np.random.rand(4) # => array([0.42, 0.65, 0.4, 0.89])
np.random.rand(4) # => array([0.96, 0.38, 0.79, 0.53])
```
for tf, 
`a = tf.random_uniform([1], seed=1)` # op level seed reset
`tf.set_random_seed(1)`              # graph level seed reset

# todo
1. how to debug python
2. how to save list, array or something else, and load them when needed.

6. learn tf serving,
https://www.tensorflow.org/serving/serving_basic
https://medium.com/epigramai/tensorflow-serving-101-pt-1-a79726f7c103
https://medium.freecodecamp.org/how-to-deploy-tensorflow-models-to-production-using-tf-serving-4b4b78d41700
