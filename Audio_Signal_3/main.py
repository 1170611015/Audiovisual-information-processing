import re
import os
import pickle
from utils import Speech, SpeechRecognizer
import pyaudio as pa
import wave
import numpy as np
'''
Attention
-----------------
> 需要安装的包在requirements中列出
> 这里没有使用 1 3 录音人的语料
'''
CATEGORY = ['0', '1', '2', '3', '4', '5', '6']  # 7 categories
test_set = {'1', '5', '9'}
block_person_set = ['1', '3']
CHANNLS = 1  # 录音的声道数
CHUNK = 1024  # 录音的块大小
SAMPLEWIDTH = 2  # 采样字节
FRAMERATE = 16000  # 录音的采样频率
RECORDTIME = 3  # 录音的时长 单位 秒（s）
FILENAME = "./RecordedVoice/recordedVoice.wav"
FILENAME_After = "./RecordedVoice_AfterEndpointDetection/recordedVoice_afterEndpointDetection.wav"
FORMAT=pa.paInt16

def loadData(dirName, train_flag = True):
    ''' 读取dirName下的所有数据，并且直接进行wav数据的mfcc特征提取 '''
    fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']
    fileList.sort()    
    speechList = []
    
    for fileName in fileList:
        pattern = re.compile(r'(\d+)_(\d+)_(\d+).wav')
        personId, categoryId, idx = pattern.match(fileName).group(1,2,3)

        if personId in block_person_set:
            continue

        if (train_flag and (not idx in test_set)) \
          or ((not train_flag) and (idx in test_set)):
            # print(fileName)
            speech = Speech(dirName, fileName)
            speech.extractFeature()
            speechList.append(speech)

    return speechList

def training(speechRecognizerList):
    ''' HMM training
    > 运行Baum-Welch算法进行模型的参数重估 '''
    for speechRecognizer in speechRecognizerList:
        speechRecognizer.trainHmmModel()

    return speechRecognizerList

def loadModel():
    print('loading all models now ...')
    speechRecognizerList = []
    
    # initialize speechRecognizer
    for categoryId in CATEGORY:
        speechRecognizer = SpeechRecognizer(categoryId)
        speechRecognizerList.append(speechRecognizer)
        speechRecognizer.initHmmModel(load_model=True)

    return speechRecognizerList

def saveModel(speechRecognizerList):
    for recognizer in speechRecognizerList:
        recognizer.saveHmmModel()

def pre_training(n_iter, speechList):
    ''' HMM pre training using viterbi
    > 首先对Gaussion HMM模型进行初始化，之后迭代运行viterbi算法对模型的初始参数进行确定'''
    speechRecognizerList = []
    
    # initialize speechRecognizer
    for categoryId in CATEGORY:
        speechRecognizer = SpeechRecognizer(categoryId)
        speechRecognizerList.append(speechRecognizer)
    
    # organize data into the same category
    for speechRecognizer in speechRecognizerList:
        # 针对一个语料使用 viterbi 算法进行 pre_training
        for speech in speechList:
            if speech.categoryId ==  speechRecognizer.categoryId:
                speechRecognizer.trainData.append(speech.features)
        
        # get hmm model
        # 当 nComp 设置不恰当的时候，运行 hmm 的过程中会出现nan的错误
        speechRecognizer.initModelParam(nComp = 12, n_iter = 15)
        speechRecognizer.stackTrainData()
        speechRecognizer.initHmmModel()

    for speechRecognizer in speechRecognizerList:
        for iter in range(n_iter):
            speechRecognizer.viterbi()

    return speechRecognizerList

def recognize(testSpeechList, speechRecognizerList):
    ''' recognition ''' 
    predictCategoryIdList = []
    
    for testSpeech in testSpeechList:
        scores = []
        
        for recognizer in speechRecognizerList:
            score = recognizer.hmmModel.score(testSpeech.features)
            scores.append(score)
        
        idx = scores.index(max(scores))
        predictCategoryId = speechRecognizerList[idx].categoryId
        predictCategoryIdList.append(predictCategoryId)

    return predictCategoryIdList


def calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList):
    ''' calculate recognition rate '''
    score = 0
    length = len(groundTruthCategoryIdList)
    
    for i in range(length):
        gt = groundTruthCategoryIdList[i]
        pr = predictCategoryIdList[i]
        
        if gt == pr:
            score += 1
    
    recognitionRate = float(score) / length
    return recognitionRate
    

def train4Models():
    ### Step.1 读取训练数据
    print('Step.1 Training data loading...')
    trainDir = 'audioRes/'
    trainSpeechList = loadData(trainDir, train_flag=True)
    print('d1one!') 
    ### Step.2 训练
    print('Step.2 Pre Training model...')
    speechRecognizerList = pre_training(7, trainSpeechList)

    print('Step.3 Training model...')
    speechRecognizerList = training(speechRecognizerList)
    print('done!')
    ### Step.3.5 Save Models 
    print('Step.3.5 Save All Models')
    saveModel(speechRecognizerList)
    ### Step.3 读取测试数据
    print('Step.3 Test data loading...')
    testDir = 'audioRes/'
    testSpeechList = loadData(testDir, train_flag=False)
    print('done!')
    ### Step.4 识别
    print('Step.4 Recognizing...')
    predictCategoryIdList = recognize(testSpeechList, speechRecognizerList)
    ### Step.5 打印结果
    groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    recognitionRate = calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList)
    
    print('===== Final result =====')
    print('Ground Truth:\t', groundTruthCategoryIdList)
    print('Prediction:\t', predictCategoryIdList)
    print('Accuracy:\t', recognitionRate)
    

def testModels():
    ### Step.3.5 Save Models 
    print('Step.3.5 Save All Models')
    speechRecognizerList = loadModel()
    ### Step.3 读取测试数据
    print('Step.3 Test data loading...')
    testDir = 'audioRes/'
    testSpeechList = loadData(testDir, train_flag=False)
    print('done!')
    ### Step.4 识别
    print('Step.4 Recognizing...')
    predictCategoryIdList = recognize(testSpeechList, speechRecognizerList)
    ### Step.5 打印结果
    groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    recognitionRate = calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList)
    
    print('===== Final result =====')
    print('Ground Truth:\t', groundTruthCategoryIdList)
    print('Prediction:\t', predictCategoryIdList)
    print('Accuracy:\t', recognitionRate)

def loadLabelName():
    labelNameList = {}
    with open('label_name.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            x, y = line.split()
            labelNameList[x] = y
    return labelNameList

def inference(recognizerList, dirName, fileName, labelNameList):
    '''
    Params
    ----------
    recognizerList: recognizer 序列, 需要提前调用loadModel进行加载
    dirName: wav文件目录地址
    fileName: 需要识别的wav文件名称
    
    返回值：(预测得到的类别Id, 对应的类别语音label)

    Description
    ------------
    对fileName指定的wav文件进行mfcc特征提取后使用HmmModel进行推测，并返回推测类别ID
    '''
    print('inference wav file {0}'.format(fileName))
    speech = Speech(dirName, fileName)
    speech.extractFeature()

    scores = []
        
    for recognizer in recognizerList:
        score = recognizer.hmmModel.score(speech.features)
        scores.append(score)
    
    idx = scores.index(max(scores))
    predictCategoryId = recognizerList[idx].categoryId
    print('\tpredict result : {0}'.format(labelNameList[predictCategoryId]))
    return predictCategoryId, labelNameList[predictCategoryId]

def recordVoice():
    start=str(input("是否录音？？Y（yes） or N（no）"))
    if start==str("Y") or start==str("yes") or start==str("y"):
        pya = pa.PyAudio()
        audio_stream = pya.open(format=FORMAT, channels=1, rate=FRAMERATE, input=True, frames_per_buffer=CHUNK)
        print("请开始讲话。。。。。。。。。。")
        frames = []
        for i in range(0, int(FRAMERATE / CHUNK * RECORDTIME)):
            data = audio_stream.read(CHUNK)
            frames.append(data)
        print("录音结束。。。。。。。。。。。")

        #检查是否有之前的录音文件，若有则删除
        if os.path.exists(FILENAME):
            os.remove(FILENAME)
        if os.path.exists(FILENAME_After):
            os.remove(FILENAME_After)

        with wave.open(FILENAME, 'wb') as f:
            f.setnchannels(nchannels=CHANNLS)
            f.setsampwidth(sampwidth=SAMPLEWIDTH)
            f.setframerate(framerate=FRAMERATE)
            f.writeframes(b"".join(frames))
            f.close()

        audio_stream.stop_stream()
        audio_stream.close()
        pya.terminate()
    else:
        frames=[]
    return frames


def sgn(data):
    if data >= 0:
        return 1
    else:
        return -1


#  计算短时能量
def calEnergy(wave_data):
    energy = []
    sum = 0
    for i in range(len(wave_data)):
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0:  # 计算每一帧的能量256个采样点为一帧
            energy.append(sum)
            sum = 0  # 计算完一帧之后将暂时和置零
        elif i == len(wave_data) - 1:  # 对于不足一帧的最后部分，直接扩充计算
            energy.append(sum)
    return energy


# 计算过零率
def calZeroCrossingRate(wave_data):
    zeroCrossingRate = []  # 存储每一帧的过零率
    sum = 0  # 各采样点的暂时和
    for i in range(len(wave_data)):
        if i % 256 == 0:
            continue
        sum = sum + np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % 256 == 0:
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1:
            zeroCrossingRate.append(float(sum) / 255)
    return zeroCrossingRate


# 利用短时能量，短时过零率，使用双门限法进行端点检测
# 找静音部分，能量小于某一门限，且连续持续若干帧
def endPointDetect(wavedata, energy, zeroCrossingRate):
    count = 0
    averageEnergy = 0
    for e in energy:
        count = count + e
    averageEnergy = count / len(energy)
    count = 0
    for e in energy[:5]:
        count = count + e
    ML = count / 5
    MH = averageEnergy / 4  # 较高的能量阈值
    ML = (ML + MH) / 4  # 较低的能量阈值
    sum1 = 0
    for zcr in zeroCrossingRate[:5]:
        sum1 = float(sum1) + zcr
    Zs = sum1 / 5  # 过零率阈值

    A = []  # 存储检测出的浊音
    B = []  # 存储检测出的第二段语音
    C = []  # 存储整个语音段

    # 利用较大能量阈值 MH 进行初步检测
    flag = 0
    for k in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[k] > MH:
            A.append(k)
            flag = 1
        elif flag == 0 and energy[k] > MH and k - 21 > A[len(A) - 1]:
            A.append(k)
            flag = 1
        elif flag == 0 and energy[k] > MH and k - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[k] < MH:
            A.append(k)
            flag = 0
    # print("计算后的浊音:" + str(A))
    # 利用较小能量阈值 ML 进行第二步能量检测
    for j in range(len(A)):
        k = A[j]
        if j % 2 == 1:
            while k < len(energy) and energy[k] > ML:
                k = k + 1
            B.append(k)
        else:
            while k > 0 and energy[k] > ML:
                k = k - 1
            B.append(k)
    # print("增加一段语音:" + str(B))
    # 利用过零率进行最后一步检测
    for j in range(len(B)):
        k = B[j]
        if j % 2 == 1:
            while k < len(zeroCrossingRate) and zeroCrossingRate[k] >= 3 * Zs:
                k = k + 1
            C.append(k)
        else:
            while k > 0 and zeroCrossingRate[k] >= 3 * Zs:
                k = k - 1
            C.append(k)
    # print("最终语音:" + str(C))
    return C


# 将语音文件存储成 wav 格式
def save_wave_file(filename, data):
    file = wave.open(filename, 'wb')
    file.setnchannels(nchannels)
    file.setsampwidth(sampwidth)
    file.setframerate(framerate)
    file.writeframes(b"".join(data))
    file.close()

if __name__ == '__main__':
    # 这里是一个调用接口的例子

    # 首先加载已经训练好的模型 
    recognizerList = loadModel()
    # 加载 labelName
    labelNameList = loadLabelName()

    rv=recordVoice()
    f = wave.open(FILENAME, "rb")
    # getparams() 一次性返回所有的WAV文件的格式信息
    params = f.getparams()
    # nframes 采样点数目
    nchannels, sampwidth, framerate, nframes = params[:4]
    # readframes() 按照采样点读取数据
    str_data = f.readframes(nframes)  # str_data 是二进制字符串
    # 转成二字节数组形式（每个采样点占两个字节）
    wave_data = np.fromstring(str_data, dtype=np.short)
    # print("采样点数目：" + str(len(wave_data)))  输出应为采样点数目
    f.close()
    energy = calEnergy(wave_data)  # 计算每一帧的短时能量
    zeroRate = calZeroCrossingRate(wave_data)  # 计算过零率
    N = endPointDetect(wave_data, energy, zeroRate)  # 利用短时能量和短时过零率进行端点检测
    m = 0
    while m < len(N) - 1:
        save_wave_file(FILENAME_After,
                       wave_data[N[m] * 256: N[m + 1] * 256])
        m = m + 2

    # 利用HmmModel进行推理，得出预测结果
    inference(recognizerList, './RecordedVoice_AfterEndpointDetection/', 'recordedVoice_afterEndpointDetection.wav', labelNameList)
