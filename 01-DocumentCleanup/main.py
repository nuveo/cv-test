#!/usr/bin/env python3

import preprocessing

import sys
import argparse
import os


parser = argparse.ArgumentParser(description='Document Cleanup.')
parser.add_argument('-p', '--path', metavar='path', type = str,
                           default = 'noisy_data', required = True,
                           help = 'Remoção de ruido das imagens na pasta "path"')


""" Document Cleanup"""

if __name__ == "__main__":

    args = parser.parse_args()
    if not os.path.isdir(args.path):
        print('The path specified does not exist.')
        sys.exit()

    preprocessing.remove_noisy(args.path)
    print("Preprocessing done!")
        