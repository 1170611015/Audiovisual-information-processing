from struct import unpack

# 读取并存储 bmp 文件

def parse_bmp(filePath):
    file = open(filePath, "rb")
    # 读取 bmp 文件的文件头    14 字节
    bfType = unpack("<h", file.read(2))[0]       # 位图文件类型
    bfSize = unpack("<i", file.read(4))[0]       # 位图文件大小
    bfReserved1 = unpack("<h", file.read(2))[0]  # 保留字段 必须设为 0
    bfReserved2 = unpack("<h", file.read(2))[0]  # 保留字段 必须设为 0
    bfOffBits = unpack("<i", file.read(4))[0]    # 偏移量
    print(bfType)

bmp_class = parse_bmp('C:\\Users\\liangmingliang\\PycharmProjects\\Visual_signal_!\\photo.png')
print(0x4d42)