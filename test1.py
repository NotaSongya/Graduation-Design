#!/usr/bin/env python
# coding: utf-8

# In[10]:


import ctypes
import math
import mss.tools
import torch
import os
from PIL import Image
from pynput.mouse import Controller
from io import BytesIO

class Point:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class Line(Point):
    def __init__(self, x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)

    def getlen(self):
        return math.sqrt(math.pow((self.x1 - self.x2), 2) + math.pow((self.y1 - self.y2), 2))

device = torch.device("cuda")
model = torch.hub.load('K:/jupterwork/yolo', 'custom',
                       'K:/jupterwork/yolo/yolov5s.pt',
                       source='local', force_reload=False)
# 定义屏幕宽高
game_width = 1920
game_height = 1080

rect = (0, 0, game_width, game_height)
m = mss.mss()
mt = mss.tools

"""
driver = ctypes.CDLL('K:/jupterwork/yolo/logitech.driver.dll')
ok = driver.device_open() == 1
if not ok:
    print('初始化失败, 未安装lgs/ghub驱动')
"""

try:
    # 获取当前绝对路径
    #root = os.path.abspath(os.path.dirname(__file__))
    root = os.getcwd()
    driver = ctypes.CDLL(f'{root}/logitech.driver.dll')
    ok = driver.device_open() == 1  # 该驱动每个进程可打开一个实例
    if not ok:
        print('错误, GHUB驱动没有找到')
except FileNotFoundError:
    print(f'错误, DLL 文件没有找到')

"""
def screen_record():
    img = m.grab(rect)
    #mt.to_png(img.rgb, img.size, 6, "temp/cfbg.png")
    mt.to_png(img, img.size, output="temp/cfbg.png")
"""
cached_image_path = 'temp/cfbg.png'
def screen_record():
    img = m.grab(rect)
    img_bytes = Image.frombytes("RGB", img.size, img.rgb)
    mt.to_png(img_bytes.tobytes(), img.size, output="temp/cfbg.png")

while True:
    screen_record()
    model = model.to(device)
    results = model(cached_image_path)
    xmins = results.pandas().xyxy[0]['xmin']
    ymins = results.pandas().xyxy[0]['ymin']
    xmaxs = results.pandas().xyxy[0]['xmax']
    ymaxs = results.pandas().xyxy[0]['ymax']
    class_list = results.pandas().xyxy[0]['class']
    confidences = results.pandas().xyxy[0]['confidence']
    newlist = []
    for xmin, ymin, xmax, ymax, classitem, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
        #if classitem == 0 and conf > 0.5:
        if conf > 0.5:
            newlist.append([int(xmin), int(ymin), int(xmax), int(ymax), conf])
    # 循环遍历每个敌人的坐标信息传入距离计算方法获取每个敌人距离鼠标的距离
    if len(newlist) > 0:
        print('newlist:', newlist)
        # 存放距离数据
        cdList = []
        xyList = []
        for listItem in newlist:
            # 当前遍历的人物中心坐标
            xindex = int(listItem[2] - (listItem[2] - listItem[0]) / 2)
            yindex = int(listItem[3] - (listItem[3] - listItem[1]) * 2 / 3)
            mouseModal = Controller()
            x, y = mouseModal.position
            L1 = Line(x, y, xindex, yindex)
            print(int(L1.getlen()), x, y, xindex, yindex)
            # 获取到距离并且存放在cdList集合中
            cdList.append(int(L1.getlen()))
            xyList.append([xindex, yindex, x, y])
        # 这里就得到了距离最近的敌人位置了
        minCD = min(cdList)
        # 如果敌人距离鼠标坐标小于150则自动进行瞄准，这里可以改大改小，小的话跟枪会显得自然些
        if minCD < 250:
            for cdItem, xyItem in zip(cdList, xyList):
                if cdItem == minCD:
                    print(cdItem, xyItem)
                    # 使用驱动移动鼠标
                    driver.moveR(int(xyItem[0] - xyItem[2]),
                                 int(xyItem[1] - xyItem[3]), True)
                break


# In[ ]:





# In[ ]:




