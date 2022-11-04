import pickle
import sys

import pandas
import pandas as pd
import os
import math
import random

pd.set_option('display.max_rows', None)
# ref)https://emilkwak.github.io/pandas-dataframe-concat-efficiently

##판다스 동시 적재
# base_dir = 'D:/Semester2201/LAB/etc/TrajectoryPredict/preprocessing/data'
#
# list_of_df = [] # 빈 list를 만듦.
# for fname in os.listdir(base_dir):
#     df_temp = pd.read_pickle(os.path.join(base_dir, fname))
#     list_of_df.append(df_temp) # 매 Loop에서 취해진 DataFrame을 위의 list에 적재함.
# df_accum = pd.concat(list_of_df).reset_index()
#
# print(df_accum.info())
# print(df_accum.head())
# print(df_accum)

'''
    일단 데이터 프레임 하나에 대해서 filtering 하기
    1. 차량 id 한 대 당 50개가 되는 경우 : [[x,y],...,[x,y]] 형태로 result df에 추가
    2. 50대 이하인 경우
        a - 20개 이하 : drop
        b - 20개 이상 : frame 기준으로 선형 보간(보간한 값이 신뢰할 수 있도록!)
          1. frame 내 결측치 보간
          2. 데이터 argumentation(보간)
               1. Max(실제값 frame)-Min(실제값 frame)
               - 50-len(실제 값)
    3. 50대 이상인 경우
        - frame이 전/후반에 몰려있는 경우, (1초에서 별로 차이가 안나서 별 상관 없을 것 같긴한데 그래도) 뽑을때, 한 쪽으로 치우칠 우려가 있어
        1. frame 내 결측치 보간
        2. 데이터를 50개의 구간으로 나누고, 구간별로 mean값을 취함... 을 생각했는데 df로 할 거면 index를 뽑음


    -> 구현 과정
    하나의 파일에서
    1. 차량 당 궤적 데이터가 20개 이하인 경우 drop
    2. frame 기준으로 결측치 보간 - 전체 데이터데 대해서
    3. 개수 기준으로 결측치 보간
    4. 50개 이상인 궤적을 50개로 sampling - sampling index 선정 : range(0, max(index), math.trunc(max(index)//50)



'''
# with open('./data/dumm.pickle', 'rb') as f :
#      data = pickle.load(f)
'''
    예제로 사용하기 위해 dataframe 합침
'''
defSeqNum = 50  # 차량 한 대당 기준이 되는 궤적 데이터 개수. 50
nameList = ["dumm", "dumm2", "dumm3"]
# nameList = ["dumm"]
list_of_df = []
for fIdx in range(len(nameList)):
    ### 피클 파일 불러오기 ###
    with open("./data/{}.pickle".format(nameList[fIdx]), "rb") as fr:
        data = pickle.load(fr)
        data = data.astype('int')
        # print(data.info())
        list_of_df.append(data)
df_accum = pd.concat(list_of_df).reset_index()

# todo 현재 코드는 파일 하나에 대한 전처리 -> 전체 데이터 개수에 대해서 변경해야 함
# 이때 고려사항 :
# # 2-a. 차량 하나 당  ['frame'] 개수가 20개 이하인 경우 drop
id_data = df_accum.groupby('id')['frame'].apply(list)  # id 별 frame 리스트를 Series로 추출
# print("id_data:", id_data) #20개 보다 작은 값이 유효함
for fIdx in range(len(id_data.index)):
    if len(id_data.iloc[fIdx]) <= 20:  # 20개 보다 작으면,
        df_accum = df_accum.drop(index=df_accum.loc[df_accum['id'] == id_data.index[fIdx], :].index)

# if (len(filledDf) <= 50):
#     if len(filledDf) <= 50:
#         df_accum = df_accum.drop[]

# 1. frame내 결측치 - frame [min:max] 값을 ['frame']으로 갖는 dataframe을 만들고, 원본과 join

# print(df_accum.groupby('id')['frame'].apply(list).index)#id 별 frame 리스트를 Series로 추출
# print(df_accum.groupby('id')['frame'].apply(list).values)#id 별 frame 리스트를 Series로 추출
id_data = df_accum.groupby('id')['frame'].apply(list)  # id 별 frame 리스트를 Series로 추출
idList = id_data.index

dfList = []
list_of_xy = []
metaList = []  # (영상 id, 영상 내 차량 id) <- list_of_xy와 개수 동일해야함. camera2의 위치값 확인을 위해서
for fIdx in range(len(idList)):  #
    metaList.append(idList[fIdx])
    print("id_data:", id_data)
    print("id_data[fIdx]: ", id_data.iloc[fIdx])
    id_data_df = df_accum.loc[(df_accum.id == idList[fIdx])]  # id 별 frame 리스트를 Series로 추출
    # frame[min:max]를 'frame'으로 갖는 dataframe
    minF = min(id_data.iloc[fIdx])
    maxF = max(id_data.iloc[fIdx])
    frameList = list(range(minF, maxF + 1))
    nDf = pd.DataFrame({"frame": frameList})
    nDf = nDf.astype('int')
    # 'frame' 기준으로 join
    filledDf = pd.merge(left=nDf, right=id_data_df, how="left", on="frame")  # ['frame'] 기준의 결측치가 없는 df 에 기존 데이터 병합
    filledDf.id = idList[fIdx]  # 결측치에 id 부여
    filledDf = filledDf.drop(['index'], axis='columns')  # 기존 결측치 있는 ['index'] 삭제
    # frame 기준으로, ['x','y'] 선형 보간
    filledDf = filledDf.interpolate(method='linear', limit_direction='forward', axis=0)  # 선형 보간
    filledDf = filledDf.astype('int')  # 형 변환
    print(filledDf.info())

    # todo - 2.b.2 - 20개 이상 50개 미만인 경우의 보간 기준이 뭡니까

    # 3. 전체 값이 50개 이상인 경우 - sampling
    fillIdx = filledDf.index.tolist()
    print(fillIdx)
    print(len(fillIdx))

    samMaxIdx = max(fillIdx)
    # samIdxList = list(range(0, samMaxIdx, [1 if math.trunc(samMaxIdx // 50) == 0 ])) #50개 이하인 경우
    step = math.trunc(samMaxIdx // defSeqNum) if math.trunc(samMaxIdx // defSeqNum) != 0 else 1
    print(step)
    samIdxList = list(range(0, samMaxIdx, step))  # 50개 이하인 경우
    samIdxList = [0, *sorted(random.sample(samIdxList[1:], 48)), samIdxList[-1]]  # 시작점과 끝점은 포함이 되도록
    # len(samIdnList)=50 개로 데이터 샘플링

    sampledDf = filledDf.iloc[samIdxList].reset_index()

    posList = []

    # todo - 영상 번호와 영상 내 idx 를 튜플형식이든 뭐든 일단 저장해야함. 일단 영상 내 차량 idx 만 저장했음
    # for i in range(defSeqNum):
    for time in range(10):
        posList.append([sampledDf['x'].iloc[time], sampledDf['y'].iloc[time]])
        print(posList)

print(metaList)







