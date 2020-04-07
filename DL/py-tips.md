### os
1. [file_i for file_i in os.listdir('folder_name') if '.jpg' in file_i and '000' in file_i]

result: [file_i] a list.

2. `os.path.join`
join folder name and file name
`print(os.path.join('fodername', 'filename'))`

`files = [os.path.join('foldername', file_i) for file_i in os.listdir('foldername') if '.jpg' in file_i]`

3. inside ipython
  - help(plt) or plt.imread? to get help

  - numpy doc
    ```
    from numpy import doc
    help(doc)
    help(np.doc.TOPICS)
    ```

4. module and version check
`python3 -c 'import mo as m; print(m.__version__)'`

`python3 -m pip install --upgrade m -U`

`sudo python3 -m pip uninstall m -U`

5. matrix shape meaning
if `[218, 178, 3]`, `[:, :, 0]` means the 1st channel all data
`plt.imshow(img[:, :, 0])` colon operator to "take every value in this dimension", meaning "give me every row, every column, and the 0th dimensoin of the color channels"

6. image dtype
  - dtype, data type, uint8, uint32, float32
  - `img.dtype` to get image data type
  - `img.astype(np.float32)` to set image data type

7. visiualize image
`img = plt.imread(file_i)`
`plt.imshow(img)`

crop and resize pic
refer to `CADL/session-0` (https://github.com/pkmital/CADL.git)  
`imresize(crop, (64, 64))`

```
from PIL import Image
square = imcrop_tosquare(img)
crop = imcrop(square, 0.2)
rsz = np.array(Image.fromarray(crop).resize((64, 64)))
plt.imshow(rsz)
```

8. difference between `np.array` and `np.ndarray`
In Python, arrays from the NumPy library, called N-dimensional arrays or the ndarray, are used as the primary data structure for representing data.

ndarray is a shorthand name for N-dimensional array, data in an ndarray is simply referred to as an array. `dtype` attribute to get data type, `shape` to return a tuple describing the length of each dimension.
[array ndarray reference](https://machinelearningmastery.com/gentle-introduction-n-dimensional-arrays-python-numpy/)

```
from numpy import empty, zeros, ones
a = empty([3, 3])
print(a)
b = zeros([3, 3])
print(b)
c = ones([3, 3])
print(c)
```

9. what is `autoencoders`  
reproduce the input at the output layer, but why? where to use this nn.
  - the goal of an autoencoder architecture is to create a representation of the input at the output layer such are as close(similar) as possible.
  - the actual use of autoencoders is for determining a compressed version of the input data with the lowest amount of loss in data. data not lost, but overall size is reduced significantly. this concept is called dimensionality reduction.
[reference](https://towardsdatascience.com/deep-autoencoders-using-tensorflow-c68f075fd1a3)
[core reference](https://www.tensorflow.org/tutorials/generative/cvae)

10. art about DL
  - make an algorithm paint an image
  - hallucinate objects in a photograph
  - generate entirely new content
  - teach a computer to read and synthesize new phrases
  - generate music with DL
<<<<<<< caeca0f66a533bd290032853ec6a367b5c47f39b
=======

11. need to look into
  - reinforcement learning
  - dictionary learning
  - probabilistic graphical models
  - bayesian methods(Bishop)
  - genetic and evolutionary algorithms

12. flatten operation(numpy)
  - np.ravel(data) or data.ravel()

13. basic tf calling sequence
1. g = tf.get_default_graph()

>>>>>>> x
