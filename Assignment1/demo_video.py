
from isort import file
import numpy as np
import cv2
import os
import cv2
import numpy as np

src_path1 = '/Users/qinkexin/学习/神经网络/期末/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_02/'
src_path2  = './test_results/'
# 获取图像大小
img = cv2.imread(src_path2+'stuttgart_02_000000_005100_leftImg8bit.png')
h,w,c = img.shape
all_files = os.listdir(src_path2)
index = len(all_files)
print("图片总数为:" + str(index) + "张")

for i in range(5100,index+5100):
    filename = 'stuttgart_02_000000_00%04d_leftImg8bit.png'%i
    img1 = cv2.imread(src_path1 + filename)
    img2 = cv2.imread(src_path2 + filename)
    # 拼接图片
    img_tmp = np.hstack((img1,img2))
    cv2.imwrite("./stack_pics/"+filename, img_tmp)



# 设置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')#MP4格式
#完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
videowrite = cv2.VideoWriter('./demo/demo2.mp4',fourcc,30,(w*2, h))#30是每秒的帧数，size是图片尺寸
# 临时存放图片的数组
img_array=[]
 
for filename in ["./stack_pics/" +'stuttgart_02_000000_00%04d_leftImg8bit.png'%i for i in range(5100,index+5100)]:
    img = cv2.imread(filename)
    if img is None:
        print(filename + " is error!")
        continue
    img_array.append(img)
# 7.合成视频 
for i in range(0,index):
 img_array[i] = cv2.resize(img_array[i],(w*2,h))
 videowrite.write(img_array[i])
 print('第{}张图片合成成功'.format(i))
print('------done!!!-------')
