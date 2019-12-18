import numpy as np
import wave
import struct
import math
params = ()

def read_from_file(filename):
    f = wave.open(filename,"rb")
    global params#将参数元组声明为全局变量
    params = f.getparams()
    dataset = f.readframes(params[3])#获得不含文件头的二进制数据
    f.close()
    dataset = np.frombuffer(dataset,dtype=np.short)#将二进制数据转化成可操作的short型数组
    return dataset

def qulification_with_8bit(value,a):
    if value > 127*a:
        value = 127.5
    elif value < -127*a:
        value = -127.5
    else:
        value = value//a + 0.5
    print(value)
    return value

def coding_with_8bit(dataset,a):
    length = len(dataset)
    decodingdataset = []
    compressdata = []
    for i in range(0,length):
        if i == 0:
            difference = dataset[0]
            codingnumber = difference
        else:
            difference = dataset[i] - decodingdataset[i-1]
            codingnumber = qulification_with_8bit(difference,a)
        compressdata.append(codingnumber)
        if i == 0:
            decodingdataset.append(compressdata[0])
        else:
            decodingdataset.append(decodingdataset[i-1] + (compressdata[i])*a)
    return compressdata,decodingdataset

def makepackage(compressdata):
    length = len(compressdata)
    w = np.int8(np.ones(length-1))
    for i in range(length-1):
        w[i] = np.int8(compressdata[i+1] - 0.5)
    return w

def save_to_file(filename,w,dataset):
    f = open(filename,'wb')
    f.write(struct.pack('h',dataset[0]))
    for i in range(len(w)):
        temp = struct.pack('b',w[i])
        f.write(temp)
    f.close()

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

def decoing_with_8bit(compressdata,a):
    for i in range(len(compressdata)):
        compressdata[i] = np.int(compressdata[i][0])
    newdataset = []
    newdataset.append(compressdata[0])
    for i in range(1,len(compressdata)):
        temp = compressdata[i] + 0.5
        newdataset.append(newdataset[i-1] + (temp)*a)
    return newdataset

def save_to_pcm(filename,newdataset):
    f = wave.open(filename,'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(params[2])
    f.writeframes(np.array(newdataset).astype(np.short).tostring())
    f.close()

def snr(dataset,newdataset):
    suma = 0
    sumb = 0
    for i in range(len(dataset)):
        suma = suma + (dataset[i]**2)/1000
        sumb = sumb + ((dataset[i] - newdataset[i])**2)/1000
    snrvalue = 10*np.log10(suma/sumb)
    print("snrvalue",snrvalue)
    return  snrvalue

filename = '1.wav'
dataset = read_from_file(filename)
print(dataset[1050:1100])
a = 200
compressdata,newdataset= coding_with_8bit(dataset,a)
print(snr(dataset,newdataset))
filename = '1_8bit.dpc'
w = makepackage(compressdata)
save_to_file(filename,w,dataset)
compressdata = readcode(filename)
newdataset = decoing_with_8bit(compressdata,a)
filename = '1_8bit.pcm'
save_to_pcm(filename,newdataset)
snr(dataset,newdataset)


