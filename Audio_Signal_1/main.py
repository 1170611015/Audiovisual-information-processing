import math
import wave
import numpy as np
import pylab as pl

def Read_File(filename):
    """
    读取.wav文件
    :param filename:.wav文件的文件名
    :return: new_audio_data是将二进制文件转换后的一个可以计算的数组,audioparams是语音文件的各项参数
    """
    f = wave.open(filename,"rb")
    audioparams = f.getparams()
    audiodata = f.readframes(audioparams[3])
    f.close()
    new_audio_data = np.frombuffer(audiodata,dtype=np.short)#short是因为这个.wav文件的声音格式是2个字节表示一个取样值
    #除去44个字节的文件头
    index = np.arange(0, 11)
    new_audio_data = np.delete(new_audio_data,index)
    return new_audio_data,audioparams

#实现sgn函数
def sgn(value):
    if value >= 0:
        return 1
    else:
        return -1


def zero_crossing_rate(new_audio_data):
    """
    计算过零率
    :param new_audio_data: 语音文件的数组表示
    :return: 一个数组，存放每个帧的过零率
    """
    zeroCrossingRate = []
    sum = 0
    for i in range(len(new_audio_data)):
        if i % 256 == 0:
            continue
        sum = sum + np.abs(sgn(new_audio_data[i]) - sgn(new_audio_data[i - 1]))
        if (i + 1) % 256 == 0:
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(new_audio_data) - 1:
            zeroCrossingRate.append(float(sum) / 255)
    return zeroCrossingRate

def calenergy(new_audio_data):
    """
    计算每一个帧的能量
    :param new_audio_data:语音文件的数组表示
    :return:一个数组，表示每个帧的能量
    """
    samplenumber = len(new_audio_data)
    framelength = 256
    framenumber = math.ceil(samplenumber/framelength)
    energyarray = []
    for i in range(framenumber):
        sum = np.longlong(0)
        frame = new_audio_data[np.arange(i*framelength,min(i*framelength+framelength,samplenumber))]
        for j in range(len(frame)):
            temp = frame[j]**2
            sum = sum + temp
        energyarray.append(sum)
    return energyarray

def write_energy(filename,energyarray):
    f = open(filename,'w')
    for i in range(len(energyarray)):
        temp = str(energyarray[i])
        f.writelines(temp)
        f.writelines("\n")
    f.close()

def write_cross_rate(filename,zero_list):
    f = open(filename,'w')
    for i in range(len(zero_list)):
        temp = str(zero_list[i])
        f.writelines(temp)
        f.writelines("\n")
    f.close()

def eliminate_mute(dataset,energyarray,energylimit,zeorCrossingRate):
    """
    消除静音
    :param dataset: 语音文件
    :param energyarray: 帧能量数组
    :param energylimit: 静音的能量上限
    :return: 消除静音后的语音文件
    """
    new_dataset = []
    i = 0
    while (i>=0) and (i < len(energyarray)-1):
        #利用能量门限提取出浊音部分端点
        if energyarray[i] > energylimit:
            startpoint = i
            j = i
            while(energyarray[j] > energylimit):
                j = j +1
            endpoint = j
            利用过零率向两侧找清音部分端点
            while(zeroCrossingRate[startpoint] > 0.3):
                startpoint = startpoint - 1
            while(zeroCrossingRate[endpoint] > 0.3):
                endpoint = endpoint + 1
            for j in range(startpoint-1,endpoint):
                for k in range(256):
                    new_dataset.append(dataset[j*256+k])
            i = endpoint
        i = i + 1
    return new_dataset

def write_mute(filename,nomute_dataset,samplefre):
    """
    将清除静音后的声音数据写进文件夹里
    :param filename: 文件夹名
    :param mute_dataset: 消除静音后的声音数据
    :param samplefre: 采样频率
    :return: 无
    """
    f = wave.open(filename,'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(samplefre)
    f.writeframes(nomute_dataset.tostring())
    f.close()
"""
filename = "1.wav"
new_audio_data,audioparams = Read_File(filename)
print(new_audio_data.shape)
zeroCrossingRate = zero_crossing_rate(new_audio_data)
print(zeroCrossingRate)
energyarray = calenergy(new_audio_data)
print(energyarray)
"""

for i in range(0,10):
    filename = str(i+1) + ".wav"
    dataset,audioparams = Read_File(filename)
    print(dataset[1000:1020])
    """
    zeroCrossingRate = zero_crossing_rate(dataset)
    energyarray = calenergy(dataset)
    print(len(energyarray))
    energylimit = 32500000#能量门限
    samplefre = audioparams[2]
    new_dataset = eliminate_mute(dataset,energyarray,energylimit,zeroCrossingRate)
    mute_dataset = np.array(new_dataset).astype(np.short)
    filenamewithenergy = str(i+1) + "_en.txt"
    filenamewithcross = str(i+1) + "_zero.txt"
    filenamewithmute = str(i+1) + ".pcm"
    write_energy(filenamewithenergy,energyarray)
    write_cross_rate(filenamewithcross,zeroCrossingRate)
    write_mute(filenamewithmute,mute_dataset,samplefre)
    """






