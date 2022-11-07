import pickle
import sys

import pandas
import pandas as pd
import os, glob
import math
import random

# row 전체 보기
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
    2. 50개 이하인 경우
        a - 20개 이하 : drop
        b - 20개 이상 : frame 기준으로 선형 보간(보간한 값이 신뢰할 수 있도록!)
          1. 프레임 개수에 맞춰서(30, 60) frame 수에 맞춰 10초로 늘려서 사용

    3. 50개 이상인 경우

    -> 구현 과정
    하나의 파일에서
    1. 차량 당 궤적 데이터가 20개 이하인 경우 drop
    2. frame 기준으로 결측치 보간 - 전체 데이터에 대해서
    3. 50개 이상인 궤적을 50개로 sampling - sampling index 선정 : range(0, max(index), math.trunc(max(index)//50)


# /mnt/hdd_02_4T/10mp4/1025wb10 <- 60frame <- 확인 필요
# /mnt/hdd_02_4T/10mp4/1025all617  <- 30frame
# mnt/hdd_02_4T/10mp4/1025all617
'''


'''
    def onePickleProcessing() : 
        하나의 pickle 파일 내 id 별로 프레임 개수를 세고, 프레임 개수가 20개가 되지 않는 경우(1초보다 적게 나오는 경우, 해당 차량은 삭제함)
    1. id 기준으로 group by
    2. 각 id에 대해 frame*10 까지의 인덱스를 갖게 한 후 join
    3. 선형보간
    4. 50개로 resample

    1-5 는 pickle 파일 하나에 대해 이뤄지고,
    2-4는 pickle 파일 내 하나의 id에 대해 이뤄지므로, 이를 반복문을 이용해 함 <- 해당 부분을 병렬 프로세싱을 할 수 있으면 더 좋을 것 같음
    
    5. 해당 데이터를 학습 데이터로 사용
'''
def onePickleProcessing(fileName, df, frameNum, defSeqNum) : 
    df_accum = df.copy()

    id_data = df_accum.groupby('id')['frame'].apply(list)  # id 별 frame 리스트를 Series로 추출
    idList = id_data.index
    df_accum = df_accum.astype('int')
    # print("id_data:", id_data) #20개 보다 작은 값이 유효함
    # 차량 하나 당  ['frame'] 개수가 20개 이하인 경우 drop
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
    list_of_xy = [] # 영상 파일명, 차량 id,[] [frame, x, y], [frame, x, y], [frame, x, y], ... , [frame, x, y]]
    
    for carIdx in range(len(idList)):  #
        frame = [i for i in range(0,frameNum*10)]
        emptyDf = pd.DataFrame({'frame' : frame}) #프레임 개수만큼 빈 데이터셋 생성
        emptyDf = emptyDf.astype('int')

        id_data_df = df_accum.loc[(df_accum.id == idList[carIdx])]  # id 별 frame 리스트를 Series로 추출
        filledDf = pd.DataFrame.merge(emptyDf, id_data_df, left_on='frame', right_on='frame', how='left')   # ['frame'] 기준 결측치가 없는 df 에 기존 데이터 병합
        filledDf = filledDf.interpolate(method='linear',limit_direction='both', axis=0)  # ['frame'] 기준으로 선형 보간
        try : 
            filledDf = filledDf.astype('int')
        except : 
            # idx = 300을 넘는 경우 10초 이상의 영상에서 추출된 이미지여서 제외함
            continue
            # print(filledDf)
            # print(filledDf.info())
            # print(id_data_df.info())
            
            # print(id_data_df.loc[].info())
            # print(id_data_df.head())
            # print()
            # print(fileName)
            # sys.exit()

        # [frame, x, y] 저장
        posList = []
        [posList.append([filledDf['frame'].iloc[sampleIdx*6],filledDf['x'].iloc[sampleIdx*6], filledDf['y'].iloc[sampleIdx*6]]) for sampleIdx in range(defSeqNum)]
        list_of_xy.append([fileName, carIdx, posList])
    
    list_Df = pd.DataFrame(list_of_xy, columns = ['videoName', 'carId', 'xyPos'])
    # print(list_Df)
    # print(list_Df.info())

    return list_Df



# with open('./data/dumm.pickle', 'rb') as f :
#      data = pickle.load(f)
'''
    예제로 사용하기 위해 dataframe 합침
'''

#HDD의 파일을 불러오기 위한 경로 설정
os.chdir("/workspace")
print(os.getcwd())
print(os.listdir())

filePath = 'mnt/hdd_02_4T/10mp4/1025wb10' #60frame
fileList = os.listdir('mnt/hdd_02_4T/10mp4/1025wb10')
# filePath = 'mnt/hdd_02_4T/10mp4/1025all617' #30frame
# fileList = os.listdir('mnt/hdd_02_4T/10mp4/1025all617')
file_list_py = [file for file in fileList if file.endswith(".pickle")]


# 함수 사용에 필요한 상수 설정
defSeqNum = 50
frameNum = 60

# 데이터 전처리
list_of_df = [] # 빈 list를 만듦.
# for fIdx in range(10):
for fIdx in range(len(file_list_py)):
    ### 피클 파일 불러오기 ###
    with open(filePath+"/{}".format(file_list_py[fIdx]), "rb") as fr:
        data = pickle.load(fr)
        data = data.astype('int')
        print(data.info())
        print(data.columns)
        fileName = file_list_py[fIdx]
       
        list_Df = onePickleProcessing(fileName, data, frameNum , defSeqNum)
        # print(data)
        list_of_df.append(list_Df)
df_accum = pd.concat(list_of_df).reset_index(drop=True)


print(df_accum)
print(df_accum.info())

df_accum.to_pickle('home/dblab/haeun/Trajectory_Pred/preprocessing/data/rawToData1107_60frame.pickle')
