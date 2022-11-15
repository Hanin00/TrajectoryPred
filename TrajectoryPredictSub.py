# ==============================================================
'''
    Trajectory Predict 개선 - 22.11.15
    이하은 hhaeun5419@gmail.com

    1. camera 1에서 포착된 차량의 궤적을 특징으로 하여 궤적 데이터를 학습함
        이때, 궤적은 10초동안 지나가는 궤적이며, 1초당 5장의 프레임 수를 가짐 
        없는 경우 데이터 전처리는 다음과 같은 방식으로 진행됨
    2. camera 1에서 수집한 궤적 데이터를 바탕으로, camera 1이 아닌 2에서 나타나는 차량의 위치를 학습함
        카메라 간 바라보고 있는 방향이 다르고, 시점이 일치하지 않기 때문에 
        이를 고려해 LSTM이 예측하는 시점인 Horizontal coefficient를 설정해야한다. 
        -> n 대에 대해 직접적으로(말고 일일히? 다른 단어) 확인했을 때, 
        평균적으로(자세한 차량 대 수 확인) 약 5초 정도의 지연이 있었음
        -> 1초당 5장의 frame을 샘플링 해 해당 차량의 위치를 특징값으로 쓰므로, 
        25frame 뒤의 차량의 위치를 예측한다. 
'''

'''
    코드 
    GPU 사용 여부 확인
    1. 전처리
        - y값(camera 2의 x, y 값)
    2. 데이터 로드
    3. 모델 아키텍쳐
        - 25프레임 뒤의 챠량(x,y 좌표)을 예측하도록 되어 있는지
    4. 학습 시 hyper parameter 정리
    5. 데이터가 많지 않으므로 500 epoch 당 모델 저장하도록
'''
# ==============================================================
import pandas as pd
import numpy as np

import matplotlib
import glob, os
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
import sys
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

from multiprocessing import Process



matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

#[전체 데이터 개수,300][2 - XY][2 - 0 : x, 1 : y]
# print(len(data.iloc[0]))
# data.iloc[0][:][0][0]
# print(data.iloc[0][:][0][1])
# print(len(data.iloc[0][:][0][0]))
# print(len(data.iloc[0][:][0][1]))
# print(len(data.iloc[0][:]))
# print(data.iloc[0][:][1])


def loadData(data) :
    x_x = data.iloc[:][0][0]
    x_y = data.iloc[:][0][1]
    y = data.iloc[:][1]

    i = 0
    data = {"x_x": x_x,
            "x_y": x_y, }
    dataPd = pd.DataFrame(data)
    dataPd.loc[len(dataPd)] = y

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataPd[['x_x', 'x_y']].values)
    reframed = series_to_supervised(scaled_data, 5,1)  # t = 50 ;  # 12 -> step = 5 + predict = 1 <- feature = x_pos, y_pos

    train_days = 50  # 50
    # valid_days = 2
    values = reframed.values
    train = values[:train_days + 1, :, ]
    # valid = values[-valid_days:, :] #<-전체 데이터에서 분류할 것
    # return values, train, valid
    return values, train, scaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]


    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim):  # num_layers : 2, hidden_dim : 32, input_dim : 1, self : LSTM(1,32,2,batch_firsttrue)
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # fc = nn.Linear(hidden_dim, output_dim)

        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop  all the way to the start even after going through another batch

        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x)

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        # out = self.fc(out[:, -1, :])
        out = self.fc(out[:, :])
        # out.size() --> 100, 10
        return out



def train(trainData) :
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
# 데이터 하나 당 epoch 씩 학습
    # for i in range(len(trainData)) :
    for i in range(1) : 

        train_values, train_data, scaler = loadData(trainData.iloc[i])
        train_X_, train_y_ = train_data[:, :-2], train_data[:, -2:]  # 끝에 두 개가  Y의 x,y에 대한 예측값


        train_X = torch.Tensor(train_X_)
        train_y = torch.Tensor(train_y_)

        print(train_X.shape, train_y.shape) #(46, 10) (46, 2)
        print("train data Num : ",i)
        for t in range(num_epochs):

            train_X = torch.Tensor(train_X)
            train_y = torch.Tensor(train_y)

            y_train_pred = model(train_X)

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
            train_predict = model(train_X)


    plt.figure(figsize=(24, 8))
    plt.xlabel('x')
    plt.ylabel('y')


    # train-values의 X값 비교
    plt.title(label="train-values의 X값 비교")
    plt.plot(list(range(len(train_values[:, 0]))), train_values[:, 0], label='raw_trajectory', c='b')
    plt.plot(list(range(len(train_predict[:, 0]))), train_predict[:, 0].detach().numpy(), label='test_predict', c='r')
    plt.legend()
    plt.show()

    plt.gca()
    # train-values의 Y값 비교
    plt.title(label="train-values의 Y값 비교")
    plt.plot(list(range(len(train_values[:, 1]))), train_values[:, 1], label='raw_trajectory', c='b')
    plt.plot(list(range(len(train_predict[:, 1]))), train_predict[:, 1].detach().numpy(), label='test_predict', c='r')
    plt.legend()
    plt.show()
    #
    # x_loss = loss_fn(train_values[:,0],train_predict[:,0])
    # y_loss = loss_fn(train_values[:,1],train_predict[:,1])
    #
    # print(x_loss)


def test(testData, device, model, loss_fn) :
    loss_fn = torch.nn.MSELoss()
 
    # 데이터 하나 당 epoch 씩 학습
    # for i in range(len(testData)):
    for i in range(1):
        test_values, test_data, scaler  = loadData(testData.iloc[i])
        test_X, test_y = test_data[:, :-2], test_data[:, -2:]  # 끝에 두 개가  Y의 x,y에 대한 예측값

        test_X = torch.Tensor(test_X)
        test_y = torch.Tensor(test_y)

        #todo - loss 가 이 위치 또는 더 상위에 있어야 하나?
        test_X = torch.Tensor(test_X).to(device)
        test_y = torch.Tensor(test_y).to(device)
        y_test_pred = model(test_X)

        loss = loss_fn(y_test_pred, test_y)
        print("test loss : ", loss.item())

        # x_loss = loss_fn(test_values[0], test_y[0])
        # y_loss = loss_fn(test_values[1], test_y[1])
        #
        # print("x_loss : ",x_loss.item())
        # print("y_loss : ",y_loss.item())

        start = time.time()
        test_predict = model(test_X)
        end = time.time()
        print("시간 : ", end-start)

    # plt.figure(figsize=(24, 8))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # # for LSTM
    # #test-values의 X값 비교
    # plt.title(label="test-values의 X값 비교")
    # plt.plot(list(range(len(test_values[:, 0]))),test_values[:, 0], label='raw_trajectory', c='b')
    # plt.plot(list(range(len(test_values[:, 0]))), test_predict[:, 0].detach().numpy(), label='test_predict', c='r')
    # plt.legend()
    # plt.show()

    # plt.gca()
    # # test-values의 Y값 비교
    # plt.title(label="test-values의 Y값 비교")
    # plt.plot(list(range(len(test_values[:, 1]))), test_values[:, 1], label='raw_trajectory', c='b')
    # plt.plot(list(range(len(test_values[:, 1]))), test_predict[:, 1].detach().numpy(), label='test_predict', c='r')
    # plt.legend()
    # plt.show()


def main() : 
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    # with open('./data/data_prof.pickle', 'rb') as f :
    # with open('./data/total_3921.pickle', 'rb') as f:
    with open('./data/total_3921_ver1Frame.pickle', 'rb') as f:
        data = pickle.load(f)

    # feature = []
    # label = []
    # for i in range(len(data)) : #3921

    #     # for datarow in data.iloc[i] : 
    #     #     print(len(data.iloc[i]))
    #     #     print(datarow)
    #     #     sys.exit()
    #     rawX = [data.iloc[i][j][1] for j in range(len(data.iloc[i]))] #50
    #     rawY = [data.iloc[i][j][2] for j in range(len(data.iloc[i]))] #50
    #     feature.append([rawX, rawY])
    #     label.append([563, 523])         
       

    # df = pd.DataFrame({'feature' : feature,
    #                     'label' : label})
    # print(df.info())
    # print(df.head())
    # df.to_pickle('./preprocessing/data/total_3921_ver1Frame.pickle')


    # sys.exit()

    # print(data.iloc[0][0][0])
    # print(data.iloc[0][1][1])
    # print(data.iloc[0][1][2])
    # print(data.info())
    # print(len(data.iloc[0][1]))
    # print(data.head())

    flag = int(len(data) * 0.7) #
    # print(flag) #2744
    
    trainData = data.iloc[:flag]
    testData = data.iloc[flag:]

    #INIT - model
    #####################
    num_epochs = 200
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    # seq_dim = look_back - 1
    input_dim = 10
    hidden_dim = 128
    num_layers = 2
    output_dim = 2

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # train(trainData)

    # torch.save(model,  './model/model_200.pt')

    model = torch.load('./model/model_200.pt').to(device)
    
    # start = time.time()
    # print(start)
    torch.multiprocessing.set_start_method('spawn')
    #병렬 프로세싱d
    # procs = []    
    # for num in range(0,5):
    #     proc = Process(target=test, args=(testData,device,model,loss_fn))
    #     procs.append(proc)
    #     proc.start()
        
    # for proc in procs:
        # proc.join()
    test(testData,device,model,loss_fn)
    # end = time.time()

    # print(end)
    # print(end-start)




if __name__ == '__main__':
    main()
