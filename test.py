import numpy
from PIL import Image
import chainer.functions as F
import chainer
import data
import matplotlib.pyplot as plt
from chainer import Variable, FunctionSet, optimizers
import sys, time
import autoencorder
n_units = 100

modelc = autoencorder.AutoEncoder(F.Linear(784, n_units), 
                     F.Linear(n_units, 784), 
                     F.relu)

N = 60000
N_test = 10000
n_epoch = 20
batchsize = 100

def draw_digit(data, row, col, n, _type):
    size = 28
    plt.subplot(row, col, n)
    Z = data.reshape(size, size)
    Z = Z[::-1, :]
    plt.xlim(0, 28)
    plt.ylim(0, 28)
    plt.pcolor(Z)
    plt.title("type=%s"%(_type), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")


def forward(x_data, train):
    x = Variable(x_data)
    return modelc(x, train=train)

# import mnist
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(numpy.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(numpy.int32)

x_train, x_test = numpy.split(mnist['data'].copy(), [N])
y_train, y_test = numpy.split(mnist['data'], [N])

optimizer = optimizers.Adam()
optimizer.setup(modelc)
train_loss = []
test_loss = []

for epoch in xrange(1, n_epoch+1):
    print('epoch', epoch)
    perm = numpy.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start_time = time.time()

    for i in xrange(0, N, batchsize):
        x = numpy.asarray(x_train[perm[i:i+batchsize]])
        t = numpy.asarray(y_train[perm[i:i+batchsize]])
        
        optimizer.update(modelc, chainer.Variable(x))
        #y, loss = forward(x, True)
        #loss.backward()
        #optimizer.update()

        train_loss.append(modelc.loss.data)
        sum_loss += modelc.loss.data * batchsize

    print 'train mean loss = {}'.format(sum_loss/N)

    end_time = time.clock()
    print 'time=%.3f' % (end_time-start_time)

org_list = []
pred_list = []
for i in numpy.random.permutation(N_test)[:100]:
    xxx = x_test[i].astype(numpy.float32)
    h1 = modelc.forwardm(Variable(xxx.reshape(1, 784)), True)
    y = modelc.forwardo(h1)
    org_list.append(x_test[i])
    pred_list.append(y)

plt.figure(figsize=(15,25))

for i in xrange(10):
    for j in xrange(10):
        idx = i*10+j
        pos = (2*i)*10+j
        draw_digit(org_list[idx], 20, 10, pos+1, "orig")

    for j in xrange(10):
        idx = i*10+j
        pos = (2*i+1)*10+j
        draw_digit(pred_list[i*10+j].data, 20, 10, pos+1, "ins")

plt.show()


