from skimage import io, transform, color
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from ela import ELA
import os
import PIL
from PIL import Image, ImageChops

# ORIG = './bringback_fake_05.jpg'
# TEMP='temp.jpg'
# 构建ELA算法函数
SCALE = 10


def ELA(ORIG=None, TEMP=None):
    TEMP = 'temp.jpg'
    original = Image.open(ORIG).convert('RGB')
    original.save(TEMP, quality=90)
    temporary = Image.open(TEMP)

    diff = ImageChops.difference(original, temporary)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    #  diff.show()
    # diff.save("haha1.jpg")

    return diff


str = 'C:/Users/Zhang15197663066/Desktop/谣言信息检测-博一工程/Twitter数据集/Allpic/*.jpg'
all_pic = io.ImageCollection(str)
print(len(all_pic))

# 把系统自带的所有png示例图片，全部转换成256*256的jpg格式灰度图

# def saveRGB(f):
#  rgb = io.imread(f) # #依次读取rgb图片
#  imgnew = rgb.convert('RGB')
#  res = ELA(rgb) # #将灰度图片大小转换为256*256
# gray = color.rgb2gray(rgb) # #将rgb图片转换成灰度图
# res = res.astype('uint8')
# res=ELA(res)
#    return imgnew
'''
path_str = 'D:/rumordetection-version.1/原数据文件-twitter和微博数据集(可以操作)/准备好的微博文本和图片及列表/rumordetect_images/*.jpg'
all_pic = io.ImageCollection(path_str, load_func=ELA)

for i in range(len(all_pic)):  # #循环保存图片
    io.imsave(
        'D:/rumordetection-version.1/原数据文件-twitter和微博数据集(可以操作)/准备好的微博文本和图片及列表/rumordetect_images/pic_rumor_ela_processing_weibo/' + 'ELA_' +
        np.str_(i) + '.jpg', all_pic[i])
'''

from PIL import Image
import os

# Set the folder path
folder_path = "C:/Users/Zhang15197663066/Desktop/谣言信息检测-博一工程/Twitter数据集/Allpic/"

from PIL import Image
import os

#folder_path = 'path/to/folder'  # replace with your folder path
output_folder = 'C:/Users/Zhang15197663066/Desktop/谣言信息检测-博一工程/Twitter数据集/folderALLPIL'  # replace with your output folder path

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(folder_path):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
        continue  # skip files that are already in jpg format

    # open the image file
    file_path = os.path.join(folder_path, file_name)
    with Image.open(file_path) as im:
        # save the image file in jpg format
        new_file_name = os.path.splitext(file_name)[0] + '.jpg'
        new_file_path = os.path.join(output_folder, new_file_name)
        im.convert('RGB').save(new_file_path, 'JPEG')




path_str = 'C:/Users/Zhang15197663066/Desktop/谣言信息检测-博一工程/Twitter数据集/Allpic/*.jpg'
all_pic = io.ImageCollection(path_str, load_func=ELA)

for i in range(len(all_pic)):
    image_path = all_pic.files[i]
    image_name = os.path.basename(image_path)
    new_image_name = "ELA_" + image_name.replace(".jpg", "") + ".jpg"
    new_image_path = os.path.join(
        'C:/Users/Zhang15197663066/Desktop/谣言信息检测-博一工程/Twitter数据集/ELA_Allpic',
        new_image_name)
    io.imsave(new_image_path, all_pic[i])
