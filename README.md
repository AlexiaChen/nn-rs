# nn-rs
A traditional neural network with Rust

# MNIST handwritten digit database

original format:
http://yann.lecun.com/exdb/mnist/

converted csv format:
https://pjreddie.com/projects/mnist-in-csv/

## MNIST in CSV

The format is:

```txt
label, pix-11, pix-12, pix-13, ... , pix-nn \n
newlabel, pix-11, pix-12, pix-13, ... , pix-nn \n
...
```

where pix-ij is the pixel in the i-th row and j-th column.

pix-nn is 28*28 = 784 pixel values in the range 0-255 gray values.

label is the digit represented by the image.

For the curious, this is the script to generate the csv files from the [original data](http://yann.lecun.com/exdb/mnist/)

```python
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "mnist_test.csv", 10000)
```

# FAQ

Q: Why need to seperate the train data set and test data set?
A: That is we want  to test before we train the model. Otherwise, we can let network to remember the training data set and get a high accuracy. But it is not a good model. So that it is normal case in machine learning to seperate the train data set and test data set.

