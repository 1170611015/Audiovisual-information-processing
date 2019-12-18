import numpy as np
from utils import Speech, SpeechRecognizer
import re
import os

CATEGORY = ['0', '1', '2', '3', '4', '5', '6']  # 7 categories
test_set = {'1', '5', '9'}
block_person_set = ['1', '3']

def loadData(dirName, train_flag=True):
    ''' 读取dirName下的所有数据，并且直接进行wav数据的mfcc特征提取 '''
    fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']
    fileList.sort()
    speechList = []

    for fileName in fileList:
        pattern = re.compile(r'(\d+)_(\d+)_(\d+).wav')
        personId, categoryId, idx = pattern.match(fileName).group(1, 2, 3)

        if personId in block_person_set:
            continue

        if (train_flag and (not idx in test_set)) \
                or ((not train_flag) and (idx in test_set)):
            print(fileName)
            speech = Speech(dirName, fileName)
            speech.extractFeature()
            speechList.append(speech)

    return speechList

def euclidlen(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def dtw_algorithm(wavedata1,wavedata2):
    len1 = len(wavedata1)
    len2 = len(wavedata2)
    i = 0
    j = 0
    distancematrix = np.empty((len1,len2))#记录代价
    route = np.empty((len1,len2))#记录路径
    #采用递推的形式实现动态规划算法
    for i in range(len1):
        for j in range(len2):
            distancea = 100000
            distanceb = 100000
            distancec = 100000
            if (i-1<0)&(j-1<0):
                distancec = 100000
            elif (i-1<0)&(j-1>=0):
                distanceb = distancematrix[i][j-1] + euclidlen(wavedata1[i],wavedata2[j])
            elif (i-1>=0)&(j-1<0):
                distancea = distancematrix[i-1][j] + euclidlen(wavedata1[i-1],wavedata2[j])
            else:
                distancea = distancematrix[i-1][j] + euclidlen(wavedata1[i-1],wavedata2[j])
                distanceb = distancematrix[i][j-1] + euclidlen(wavedata1[i],wavedata2[j-1])
                distancec = distancematrix[i-1][j-1] + euclidlen(wavedata1[i-1],wavedata2[j-1])*2
            distancematrix[i][j] = min(distancea,distanceb,distancec)
            #记录回溯路径
            if (distancea<=distanceb) & (distancea<=distancec):
                route[i][j] = 0
            elif (distanceb<=distancea) & (distanceb<=distancec):
                route[i][j] = 1
            else:
                route[i][j] = 2
            #初始化时距离即为两个矢量序列的最初一个帧的距离
            distancematrix[0][0] = euclidlen(wavedata1[0],wavedata2[0])
    route[0][0] = -1
    return distancematrix[len1-1][len2-1],route

def identi_audio(wavedata,template):
    mindistance = 10000
    k = 10000
    #0-48是一个完整的模板集合，含有7段语音，每段语音重复7遍
    for i in range(0, 7):
        tempdistance, route = dtw_algorithm(wavedata,template[i])
        if tempdistance < mindistance:
            mindistance = tempdistance
            k = i
    print("k = ",k)
    return k #返回匹配最好的语音段的种类

def test():
    #测试正确率
    accuratenumber = 0
    #用49个数据做测试集
    for i in range(49,97):
        audio_index = identi_audio(trainSpeechList[i].features,template)
        if (i-49)//7 == audio_index:#如果预测结果与实际标记相匹配
            accuratenumber = accuratenumber + 1
    return accuratenumber

trainDir = 'audioRes/'
trainSpeechList = loadData(trainDir, train_flag=True)
"""
distance,route = dtw_algorithm(trainSpeechList[1].features,trainSpeechList[49].features)
print("distance = ",distance)
print(np.array(route).shape)
i = len(trainSpeechList[1].features) - 1
j = len(trainSpeechList[49].features) - 1
while (i+j>0):
    print("i= ",i," j= ",j)
    if route[i][j] == 0:
        i = i - 1
    elif route[i][j] == 1:
        j = j - 1
    else:
        i = i - 1
        j = j - 1
"""
template = []
#生成模板
for i in range(0,7):
    template.append(trainSpeechList[2+i*7].features)
audio_index = identi_audio(trainSpeechList[0].features,template)
print(audio_index)
accuratenumber = test()
print("accurate number = ",accuratenumber)
mindistance = 10000;
k = 10000;
for i in range(0,48):
    tempdistance,route = dtw_algorithm(trainSpeechList[80].features,trainSpeechList[i].features)
    if tempdistance<mindistance:
        mindistance = tempdistance
        k = i
print("mindistance = ",mindistance)
print("k = ",k)
