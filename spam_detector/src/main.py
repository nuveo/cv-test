#!/usr/bin/env python3

import inference
import train

import sys
import argparse
import os


parser = argparse.ArgumentParser(description='SMS Spam Detector.')
parser.add_argument('-p', '--path', metavar='path', type = str,
                           default = 'data/sms-hamspam-test.csv', required = True,
                           help = 'Execução do treinamento usando os dados do csv fornecido.')
parser.add_argument('-t', '--type', metavar='type', type = str,
                           default = 'inference', required = True,
                           help = 'Execução da inferência em cada linha do csv fornecido.')

""" SMS Spam Detector
Make inferences in csv files with one column. 
And can be retrained with a new dataset.
"""

if __name__ == "__main__":

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print('The path specified does not exist.')
        sys.exit()

    if args.type == "train":
        train.train(args.path)
        print("Train finished!")
    else:
        inference.inference(args.path)
        print("Inference finished!")
        