#!/usr/bin/python3

import dataset as ds
import neuralnet as NNet

eps = 0.05


def main(data):
    nn = NNet.NeuralNet(64, 56, 20)
    nn.set_param(0.4, 0.3)
    ev = eps + 1
    while( eps < ev ):
        nn.learn(data)
        ev = nn.evaluate(data)
        print(ev)

    print(nn.forward(ds.ldata_a[0][0]))
    print(nn.test(ds.ldata_a))
    print(nn.test(ds.tdata_a))
    print(nn.test(ds.ldata_b))
    print(nn.test(ds.tdata_b))
    print(nn.test(ds.ldata))
    print(nn.test(ds.tdata))


if __name__== '__main__':

    ds.load()

    # writer A
    main(ds.ldata_a)
    main(ds.ldata_b)
    main(ds.ldata)


