import pickle
'''
X data의 구간


2차선
출발점(정문 근처 횡단보도)
    x  1017 - 895
    y  222 - 335
    -> 데이터 프레임 뽑아서 랜덤하게 뽑기

일반통행 페인트
    x  793 - 944
    y  408 - 868


1차선
출발점(정문 근처 횡단보도)
    x  1083 - 941
    y  224 - 451
    -> 데이터 프레임 뽑아서 랜덤하게 뽑기

일반통행 페인트
    x  938 - 983
    y  454 - 840

Y Label의 구간
 x : 566-550
 y : 545-640
'''
import sys

import pandas as pd
import random

''' dummy y '''
datanum = 300

y_x = list(range(550,566))
extendY_x = y_x[:]
[extendY_x.extend(y_x) for i in range(23)]
extendY_x.sort(reverse=True)
y_y = list(range(454,840))

Y_data = {'x' : extendY_x,
          'y' : y_y[:len(extendY_x)]}

y_df = pd.DataFrame(Y_data)
y_xy = y_df.sample(n=datanum).values.tolist()
y_xy_list = []



[y_xy_list.append([y_xy[i][0], y_xy[i][1]]) for i in range(datanum)]
# print(y_xy_list[0])
# print(y_xy_list[49])

#x data
x_x_1_entrance = list(range(895,1017))  # reverse 필요 1017이 정문쪽 895가 교차로
x_y_1_entrance = list(range(206, 454))

x_x_1_entrance.extend(x_x_1_entrance)
x_x_1_entrance = sorted(x_x_1_entrance, reverse=True)

X_data_1 = {'x' : x_x_1_entrance,
          'y' : x_y_1_entrance[:len(x_x_1_entrance)]}


dummX = []
for i in range(datanum) :
    x_df_1 = pd.DataFrame(X_data_1)
    x_df_1 = x_df_1.sample(n=50).sort_index()

    dummX.append(x_df_1.values.tolist())


trainDummData = {'feature' : dummX,
               'label' : y_xy_list}
trainDummDf = pd.DataFrame(trainDummData)


print(trainDummDf.head())


#
# for i in range(datanum) :
#     x_df_1 = pd.DataFrame(X_data_1)
#     dfDumm = x_df_1.sample(n=50).sort_index()
#     xList = dfDumm['x'].values.tolist()
#     yList = dfDumm['y'].values.tolist()
#
#     dummX.append([xList, yList])
#
# trainDummData = {'feature' : dummX,
#                'label' : y_xy_list}
# trainDummDf = pd.DataFrame(trainDummData)
#


# save
with open('data_prof.pickle', 'wb') as f:
    pickle.dump(trainDummDf, f, pickle.HIGHEST_PROTOCOL)





# #x data
# x_x_1_bump = list(range(773,963))  # reverse 필요 X 773이 정문쪽, 963이 일반통행 페인트쪽
# x_y_1_bump = list(range(473, 870))