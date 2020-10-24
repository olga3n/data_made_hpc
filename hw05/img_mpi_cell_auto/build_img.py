#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, 'r') as fin:
        data = fin.readlines()
    
    for i in range(len(data)):
        data[i] = [int(x) for x in data[i].strip()]

    plt.imshow(np.array(data), cmap='hot', interpolation='nearest')
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
