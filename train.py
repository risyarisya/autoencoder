#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import reporter
import numpy
import matplotlib.pyplot as plt

# Network definition
pred_list = []

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

class ConvolutionalAutoencoder(chainer.Chain):
    def __init__(self, n_in, n_out, ksize, stride=1, pad=0, wscale=1, bias=0, nobias=False):
        super(ConvolutionalAutoencoder, self).__init__(
            l1 = F.Convolution2D(n_in, n_out, ksize, stride=stride, pad=pad, wscale=wscale, bias=bias, nobias=nobias),
            l2 = F.Convolution2D(n_out, n_in, ksize, stride=stride, pad=pad, wscale=wscale, bias=bias, nobias=nobias))

    def __call__(self, x, t):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        loss = F.mean_squared_error(y, t)
        reporter.report({'loss': loss}, self)
        pred_list.append(y.data[0])
        return loss

class AutoEncoder(chainer.Chain):
    def __init__(self, n_in, n_units):
        super(AutoEncoder, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_in),
        )

    def __call__(self, x, t):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        loss = F.mean_squared_error(y, t)
        reporter.report({'loss': loss}, self)
        pred_list.append(y.data[0])
        return loss

def main():
    parser = argparse.ArgumentParser(dscsription='AutoEncoder')
    parser.add_argument('--model', '-m', default='AutoEncoder',
                        help='AutoEncoder or ConvolutionalAutoencoder')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train_, test_ = chainer.datasets.get_mnist(withlabel=False)

    if args.model=='CovolutionalAutoencoder':
        model = ConvolutionalAutoencoder(1, 20, 5, pad=2)
        train_ = train_.reshape(train_.shape[0], 1, 28, 28)
        test_ = test_.reshape(test_.shape[0], 1, 28, 28)
    else:
        model = AutoEncoder(784, args.unit)

    model.compute_accuracy = False
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = tuple_dataset.TupleDataset(train_, train_)
    test = tuple_dataset.TupleDataset(test_, test_)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss']))

    trainer.extend(extensions.ProgressBar())

    trainer.run()

    for i in xrange(args.epoch):
        draw_digit(pred_list[i*700], 5, 4, i+1, 'decode')
    plt.show()

#    chainer.serializers.save_npz('model_final', model)
#    chainer.serializers.save_npz('optimizer_final', optimizer)


if __name__ == '__main__':
    main()
