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


matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

def test(testX, testY, model) :
    # 데이터 하나 당 epoch 씩 학습
    for i in range(1):
        test_X = torch.Tensor(testX)
        test_y = torch.Tensor(testY)

        #todo - loss 가 이 위치 또는 더 상위에 있어야 하나?
        test_X = torch.Tensor(test_X).to(device)
        test_y = torch.Tensor(test_y).to(device)
        y_test_pred = model(test_X)

        loss = loss_fn(y_test_pred[-1], test_y[-1])
        print("test loss : ", loss.item())
        return loss.item()
        # test_predict = model(test_X)


# 일반적인 sequential data로 변환 - 한 개 df에 대해 
# 각각 normalize 하면 denormalize 가 힘들어서 전체 값에서 normalize 함
def split_seq(seq,window,horizon,scaler_):

    df = pd.DataFrame({"x" : seq[0], "y" : seq[1]})
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_.fit_transform(df[['x','y']].values)
    df = scaled_data

    X=[]; Y=[]
    for i in range(len(seq[0])-(window+horizon)+1):
        x=df[i:(i+window)]
        y=df[i+window+horizon-1]

        # x=df.iloc[i:(i+window)]
        # y=df.iloc[i+window+horizon-1]
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

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

def train(trainX, trainY, model) : 
  for t in range(1) : 
  # for t in range(num_epochs): #궤적 데이터 하나에 대한 epoch 
    trainX = torch.Tensor(trainX)
    trainY = torch.Tensor(trainY)
    # print(trainX.shape, trainY.shape) #torch.Size([41, 5, 2]) torch.Size([41, 2])

    y_train_pred = model(trainX)

    loss = loss_fn(y_train_pred, trainY)

    # x_loss = loss_fn(y_train_pred[:, 0], trainY[:, 0])
    # y_loss = loss_fn(y_train_pred[:, 1], trainY[:, 1])
    # print("Epoch ", t, "MSE: ", loss.item())
    # if t % 10 == 0 and t != 0:
    #     print("Epoch ", t, "MSE: ", loss.item())
        # print("x_loss : ", x_loss.item())
        # print("y_loss : ", y_loss.item())

    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

  return model, loss



if __name__ == '__main__':

    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    with open('./data/xyposList1120.pickle', 'rb') as f:    
        data = pickle.load(f)

    window = 49 #며칠 전의 값 참고? # 마지막 프레임
    horizon = 1 #얼마나 먼 미래? #마지막 프레임의 위치 예측

    num_epochs = 200
    hist = np.zeros(num_epochs)

    flag = int(len(data) * 0.7) # 

    trainData = data[:flag] # 2744
    testData = data[flag:] # 1173

    input_dim = 2
    hidden_dim = 128
    num_layers = 4
    output_dim = 2
    
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    scaler_ = MinMaxScaler(feature_range=(0, 1))
    #train - 2735
    trainData = trainData[:40]
    for ep in range(num_epochs) : 
        print("epoch : ", ep)
        print("epoch : ", ep)
        print("epoch : ", ep)
        totalLoss = 0
        for idx, row in enumerate(trainData) :
            trainX, trainY = split_seq(row, window, horizon,scaler_)
            model, loss = train(trainX, trainY, model)
            totalLoss+=loss.item()
        print("total Loss mean : ",totalLoss/len(trainData[0]))

    # todo 모델 저장
    torch.save(model,  './model/term_car1_model_e200_d40.pt')

    model = torch.load('./model/term_car1_model_e200_d40.pt').to(device)
    totalLoss = 0
    for idx, row in enumerate(testData) :
      testX, testY = split_seq(row, window, horizon,scaler_)  
      totalLoss += (test(testX, testY, model))
    
    print("totalLoss mean : ", totalLoss/len(testData))
      


    
    # start = time.time()
    # print(start)

    # test(testData)
    # end = time.time()

    # print(end)
    # print(end-start)