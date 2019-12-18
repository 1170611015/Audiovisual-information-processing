import numpy as np
import wave
import struct
import math
params = ()

#读文件
def read_from_file(filename):
    f = wave.open(filename,"rb")
    global params#将参数元组声明为全局变量
    params = f.getparams()
    dataset = f.readframes(params[3])#获得不含文件头的二进制数据
    f.close()
    dataset = np.frombuffer(dataset,dtype=np.short)#将二进制数据转化成可操作的short型数组
    return dataset

#量化函数
def qulification_with_8bit(value):
    if value > 127:
        value = 127
    elif value < -127:
        value = -127
    else:
        value = value
    return value

#8bit的编码函数
def coding_with_8bit(dataset):
    length = len(dataset)
    decodingdataset = []
    compressdata = []
    for i in range(0,length):
        if i == 0:
            difference = dataset[0]
            codingnumber = difference
        else:
            #计算差值，用当前的值与上一次的解码值
            difference = dataset[i] - decodingdataset[i-1]
            #量化
            codingnumber = qulification_with_8bit(difference)
        compressdata.append(codingnumber)
        if i == 0:
            decodingdataset.append(compressdata[0])
        else:
            decodingdataset.append(decodingdataset[i-1] + compressdata[i])
        print(decodingdataset[i])
    return compressdata

#打包数据
def makepackage(compressdata):
    length = len(compressdata)
    w = np.int8(np.ones(length-1))
    for i in range(length-1):
        w[i] = np.int8(compressdata[i+1])
    return w

#写到文件里
def save_to_file(filename,w,dataset):
    f = open(filename,'wb')
    #2字节首部表示最初一个采样点
    f.write(struct.pack('h',dataset[0]))
    for i in range(len(w)):
        temp = struct.pack('b',w[i])
        f.write(temp)
    f.close()

#从文件里读出语音数据
def readcode(filename):
    code = []
    with open(filename,'rb') as f:
        a = struct.unpack('h',f.read(2))
        code.append(a)
        while True:
            flag = f.read(1)
            if not flag:
                break
            else:
                a = struct.unpack('b',flag)
                code.append(a)
    return code

#将从文件里读出的压缩数据解码
def decoing_with_8bit(compressdata):
    for i in range(len(compressdata)):
        compressdata[i] = np.int(compressdata[i][0])
    newdataset = []
    newdataset.append(compressdata[0])
    for i in range(1,len(compressdata)):
        newdataset.append(newdataset[i-1] + compressdata[i])
    return newdataset

#将解码后的数据存储成音频文件
def save_to_pcm(filename,newdataset):
    f = wave.open(filename,'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(params[2])
    f.writeframes(np.array(newdataset).astype(np.short).tostring())
    f.close()

#计算snr信噪比
def snr(dataset,newdataset):
    suma = 0
    sumb = 0
    for i in range(len(dataset)):
        suma = suma + (dataset[i]**2)/1000
        sumb = sumb + ((newdataset[i] - dataset[i])**2)/1000
    snrvalue = 10*np.log10(suma/sumb)
    return  snrvalue

filename = '1.wav'
dataset = read_from_file(filename)
compressdata = coding_with_8bit(dataset)
filename = '1_8bit.dpc'
w = makepackage(compressdata)
save_to_file(filename,w,dataset)
compressdata = readcode(filename)
newdataset = decoing_with_8bit(compressdata)
filename = '1_8bit.pcm'
save_to_pcm(filename,newdataset)
print(snr(dataset,newdataset))


