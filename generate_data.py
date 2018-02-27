import os
import numpy as np
import pickle
import argparse

desc = 'generate data'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--sample', type=int, default=10000,
                    help='The number of samples (default: 10000)')
parser.add_argument('--path', type=str, default='./data',
                    help='Data path (default: ./data)')
parser.add_argument('--filename', type=str, default='data.pkl',
                    help='File name (default: data.pkl)')
opt = parser.parse_args()


def sin(X, N, freq=100):
    w = np.random.choice(np.arange(1, 11), N,replace=False)
    sin_ = 0
    for i_w in w:
        sin_ += np.sin(i_w * np.pi * X / freq)
    sin_ = sin_/max(sin_)
    return sin_

def sample(sample_size):
    sin_ = sin(np.arange(sample_size), 3)
    noise = np.random.uniform(-0.2, 0.2, sample_size)
    return sin_ + noise

def main():
    data = np.array([sample(opt.sample) for i in range(10)])

    path = os.path.join(opt.path, opt.filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()


