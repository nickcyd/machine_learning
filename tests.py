# 测试__len__()
# l = [1,2,3]
# print(l.__len__())

# skimage图片二值化
# from skimage import io, data, color
#
# img = data.coffee()
# # img_name = '5.jpg'
# # img = io.imread(img_name, as_grey=False)
#
# img_gray = color.rgb2gray(img)
#
# rows, clos = imggray.shape
# for i in range(rows):
#     for j in range(cols):
#         if img_gray[i, j] <= 0.5:
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
# io.imshow(img_gray)
# io.imshow

# pillow 模块图片二值化
# from PIL import Image
#
# im = Image.open('5.jpg')
#
# lim = im.convert('L')  # RGB 转为灰度图
# lim.save('5L.jpg')
# lim = im.convert('1')  # RGB 转为二值化图
# lim.save('51.jpg')

# RGB 转为二值化图(自己控制像素的阈值)
# threshold = 185
# table = []
# for i in range(256):
#     if i < threshold:
#         table.append(0)
#     else:
#         table.append(1)
# bim = lim.point(table, '1')
# bim.save('511.jpg')

# 图片转 .txt 文件
# import numpy as np
# from PIL import Image
# from pylab import *
#
# img = Image.open('5.jpg')
#
# # RGB 转为二值化图
# lim = img.convert('1')
# lim.save('51.jpg')
#
# img = Image.open('51.jpg')
#
# # 将图像转化为数组并将像素转换到0-1之间
# img_ndarray = np.asarray(img, dtype='float64') / 256
#
# # 将图像的矩阵形式转化成一位数组保存到 data 中
# data = np.ndarray.flatten(img_ndarray)
#
# # 将一维数组转化成矩阵
# A = np.array(data).reshape(32, 32)
#
# # 将矩阵保存到 txt 文件中转化为二进制0，1存储
# savetxt('5_old.txt', A, fmt="%.0f", delimiter='')
#
# # 把 .txt 文件中的0和1调换
# with open('5_old.txt', 'r') as fr:
#     data = fr.read()
#     data = data.replace('1','2')
#     data = data.replace('0','1')
#     data = data.replace('2','0')
#
#     with open('5.txt', 'w') as fw:
#         fw.write(data)

# 改变图片尺寸
# from PIL import Image

# infile = '3.jpg'
# outfile = '33.jpg'
# img = Image.open(infile)
# out = img.resize((32, 32), Image.ANTIALIAS)  # resize image with high-quality
# out.save(outfile)
# print(out.size)

# 修改图片的路径
# new_img_filename = '3.jpg'
# img_filename = 'new.jpg'

# 调整图片的大小为 32*32px
# img = Image.open(new_img_filename)
# out = img.resize((32, 32), Image.ANTIALIAS)
# out.save(img_filename)
