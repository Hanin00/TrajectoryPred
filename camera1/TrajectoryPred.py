## 카메라 한 대, 차량 전체 차량에 대해 9초 후 10초때의 위치 예측
import pandas as pd
import numpy as np

import matplotlib
import glob, os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
import random

from pylab import mpl, plt

from datetime import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import gc
import torch


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False



# 일반적인 sequential data로 변환 - 한 개 df에 대해 
# 각각 normalize 하면 denormalize 가 힘들어서 전체 값에서 normalize 함
def split_seq(seq,window,horizon):
    scaler_ = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_.fit_transform(seq[['x','y']].values)

    X=[]; Y=[]
    for i in range(len(scaled_data)-(window+horizon)+1):
        x=scaled_data[i:(i+window)]
        y=scaled_data[i+window+horizon-1]
        # x=df.iloc[i:(i+window)]
        # y=df.iloc[i+window+horizon-1]
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y), scaler_

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1])
        return out


def dataloader () : 
  return tra


def train(trainX, trainY, model, num_epochs) : 
  loss_fn = torch.nn.MSELoss()
  optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

  for t in range(num_epochs):
    train_X = torch.Tensor(trainX).to(device)
    train_y = torch.Tensor(trainY).to(device)

    y_train_pred = model(train_X).to(device)

    loss = loss_fn(y_train_pred, train_y)

    x_loss = loss_fn(y_train_pred[:, 0], train_y[:, 0])
    y_loss = loss_fn(y_train_pred[:, 1], train_y[:, 1])

    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
        print("x_loss : ", x_loss.item())
        print("y_loss : ", y_loss.item())

    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()
    train_predict = model(train_X).to(device)

  return model, loss

def test(testX, testY, model) :
  # 데이터 하나 당 epoch 씩 학습
  
  loss_fn = torch.nn.MSELoss()
  test_X = torch.Tensor(testX).to(device)
  test_y = torch.Tensor(testY).to(device)

  y_test_pred = model(test_X)

  loss = loss_fn(y_test_pred, test_y)
  print("test loss : ", loss.item())

  # test_predict = model(test_X)


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    with open('./data/merged_1121.pickle', 'rb') as f:    
      data = pickle.load(f)

    window = 49 #며칠 전의 값 참고? # 마지막 프레임
    horizon = 1 #얼마나 먼 미래? #마지막 프레임의 위치 예측

    data = data[:50*300]

    num_epochs = 2000
    hist = np.zeros(num_epochs)

    flag = int(len(data) * 0.7) # 

    trainData = data.iloc[:flag] # 2744
    testData = data.iloc[flag:] # 1173

    print(flag)
    print(len(trainData))
    print(len(testData))

    input_dim = 2
    hidden_dim = 128
    num_layers = 4
    output_dim = 2
    
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)

    trainX, trainY, scaler = split_seq(trainData, window, horizon)
    model, loss = train(trainX, trainY, model, num_epochs)

    # todo 모델 저장
    torch.save(model,  './model/term_car1_model_e2000_d1500.pt')

    model = torch.load('./model/term_car1_model_e2000_d1500.pt').to(device)


    testX, testY, scaler = split_seq(testData, window, horizon)  
    test(testX, testY, model)
    
    # start = time.time()

    # print(start)

    # test(testData)
    # end = time.time()

    # print(end)
    # print(end-start)