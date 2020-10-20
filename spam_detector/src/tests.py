#!/usr/bin/env python3

import inference
import train

import inference

def test_inference():
    # "mock" data
    inference("data/sms-hamspam-test.csv")


    # confere se o csv output_inference.csv foi gerado
    # confere se tem duas colunas
    # confere se os valores são apenas ham e spam na primeira coluna
    pass


def train_test():
    # confere se tá sendo gerado o modelo 
    # confere os hyperparameters
    pass