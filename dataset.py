import numpy as np
import os

ldata_a = np.empty((20, 100, 64), dtype='float32')
tdata_a = np.empty((20, 100, 64), dtype='float32')

ldata_b = np.empty((20, 100, 64), dtype='float32')
tdata_b = np.empty((20, 100, 64), dtype='float32')

ldata = np.empty((20, 200, 64), dtype='float32')
tdata = np.empty((20, 200, 64), dtype='float32')

def print_onedata(data, ch, n):
    for i in range(8):
        for j in range(8):
            print(str(int((data[ch][n][8*i+j])*64)).zfill(2), end='')
        print()


def print_feature(feature):
    for i in range(8):
        for j in range(8):
            print(str(int(feature[8*i+j]*64)).zfill(2), end='')
        print()


def partially_feature(p, ii, jj):
    feature = 0.0
    for i in range(8):
        for j in range(8):
            if(p[8*ii + i][8*jj + j] == '1'):
                feature += 1
    return feature / 64

def extract_feature(p):
    buf = np.empty(64, dtype='float32')
    for i in range(8):
        for j in range(8):
            buf[8*i + j] = partially_feature(p, i, j)
    return buf

def read_onefile(path):
    buf = np.empty((100,64), dtype='float32')
    with open(path) as f:
        p = np.array(f.readlines())
    for charas in range(100):
        buf[charas][:] = extract_feature(p[64*charas:64*charas+64])
    return buf

def read_onewriter(path, writer):

    ldata = np.empty((20, 100, 64), dtype='float32')
    tdata = np.empty((20, 100, 64), dtype='float32')
    path = path + str(writer) + '_'
    for c in range(20):
        tmp_path = path + str(c).zfill(2) + 'L.dat'
        ldata[c][:] = read_onefile(tmp_path)
        tmp_path = path + str(c).zfill(2) + 'T.dat'
        tdata[c][:]= read_onefile(tmp_path)
    return (ldata, tdata)

def read(path='./Data'):
    path = path + '/hira'
    (ldata_a[:], tdata_a[:]) = read_onewriter(path, 0)
    (ldata_b[:], tdata_b[:]) = read_onewriter(path, 1)

def load(fname='./dataset', rpath='./Data'):
    try:
        ldata_a[:] = load_from(fname+'0l.npy')
        tdata_a[:] = load_from(fname+'0t.npy')
        ldata_b[:] = load_from(fname+'1l.npy')
        tdata_b[:] = load_from(fname+'1t.npy')
    except:
        print("faild to open serialized dataset")
        read(rpath)
        save(fname)
    for c in range(20):
        ldata[c][0:100] = ldata_a[c]
        tdata[c][0:100] = tdata_a[c]
        ldata[c][100:200] = ldata_b[c]
        tdata[c][100:200] = tdata_b[c]

def save(filename='./dataset'):
    np.save( filename+'0l.npy', ldata_a.flatten().flatten())
    np.save( filename+'0t.npy', tdata_a.flatten().flatten())
    np.save( filename+'1l.npy', ldata_b.flatten().flatten())
    np.save( filename+'1t.npy', tdata_b.flatten().flatten())

def load_from(filename):
    p = np.empty((20, 100, 64), dtype='float32')
    d = np.load(filename)
    for i in range(20):
        for j in range(100):
            p[i][j][:] = d[(64*(j+100*i)):(64*(j+100*i+1))]
    return p



