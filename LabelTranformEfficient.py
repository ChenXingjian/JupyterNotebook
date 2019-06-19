import numpy as np
import cv2
import os

raw_class=21

center_maskcolor=np.zeros((raw_class,3))
center_maskcolor[0]=[180,130,170]

center_maskcolor[1]=[128,64,128]
center_maskcolor[2]=[232,35,244]

center_maskcolor[3]=[70,70,70]

center_maskcolor[4]=[60,20,220]
center_maskcolor[5]=[0,0,255]
center_maskcolor[6]=[100,0,255]
center_maskcolor[7]=[200,0,255]
center_maskcolor[8]=[0,192,0]

center_maskcolor[9]=[35,142,107]

center_maskcolor[10]=[142,0,0]
center_maskcolor[11]=[100,60,0]
center_maskcolor[12]=[0,0,90]
center_maskcolor[13]=[32,11,119]
center_maskcolor[14]=[230,0,0]
center_maskcolor[15]=[70,0,0]
center_maskcolor[16]=[110,0,0]
center_maskcolor[17]=[192,0,0]
center_maskcolor[18]=[64,64,128]
center_maskcolor[19]=[100,80,0]

center_maskcolor[20]=[255,255,255]

# 这里使用mask的方法去处理一张图，获取某个类别的所有像素，效率比较高
# Mask method
def getkeepmask(i):
    if i==1 or i==3 or i==5 or i==7:
        if i==1:
            centermaskcolor=center_maskcolor[0]
        if i==3:
            centermaskcolor=center_maskcolor[3]
        if i==5:
            centermaskcolor=center_maskcolor[9]
        if i==7:
            centermaskcolor=center_maskcolor[20]
        
        keepmask0=np.logical_and(maskfile[:,:,0]>=(centermaskcolor[0]-5),maskfile[:,:,0]<=(centermaskcolor[0]+5))
        keepmask1=np.logical_and(maskfile[:,:,1]>=(centermaskcolor[1]-5),maskfile[:,:,1]<=(centermaskcolor[1]+5))
        keepmask2=np.logical_and(maskfile[:,:,2]>=(centermaskcolor[2]-5),maskfile[:,:,2]<=(centermaskcolor[2]+5))
        
    if i==2:
        keepmask0=np.logical_or(np.logical_and(maskfile[:,:,0]>=(center_maskcolor[1][0]-5),maskfile[:,:,0]<=(center_maskcolor[2][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[2][0]-5),maskfile[:,:,0]<=(center_maskcolor[2][0]+5)))
        keepmask1=np.logical_or(np.logical_and(maskfile[:,:,1]>=(center_maskcolor[1][1]-5),maskfile[:,:,1]<=(center_maskcolor[2][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[2][1]-5),maskfile[:,:,1]<=(center_maskcolor[2][1]+5)))
        keepmask2=np.logical_or(np.logical_and(maskfile[:,:,2]>=(center_maskcolor[1][2]-5),maskfile[:,:,2]<=(center_maskcolor[2][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[2][2]-5),maskfile[:,:,2]<=(center_maskcolor[2][2]+5)))
        
    if i==4:
        keepmask0=np.logical_or.reduce([np.logical_and(maskfile[:,:,0]>=(center_maskcolor[4][0]-5),maskfile[:,:,0]<=(center_maskcolor[4][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[5][0]-5),maskfile[:,:,0]<=(center_maskcolor[5][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[5][0]-5),maskfile[:,:,0]<=(center_maskcolor[5][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[6][0]-5),maskfile[:,:,0]<=(center_maskcolor[6][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[6][0]-5),maskfile[:,:,0]<=(center_maskcolor[6][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[7][0]-5),maskfile[:,:,0]<=(center_maskcolor[7][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[8][0]-5),maskfile[:,:,0]<=(center_maskcolor[8][0]+5))])
        keepmask1=np.logical_or.reduce([np.logical_and(maskfile[:,:,1]>=(center_maskcolor[4][1]-5),maskfile[:,:,1]<=(center_maskcolor[4][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[5][1]-5),maskfile[:,:,1]<=(center_maskcolor[5][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[5][1]-5),maskfile[:,:,1]<=(center_maskcolor[5][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[6][1]-5),maskfile[:,:,1]<=(center_maskcolor[6][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[6][1]-5),maskfile[:,:,1]<=(center_maskcolor[6][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[7][1]-5),maskfile[:,:,1]<=(center_maskcolor[7][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[8][1]-5),maskfile[:,:,1]<=(center_maskcolor[8][1]+5))])
        keepmask2=np.logical_or.reduce([np.logical_and(maskfile[:,:,2]>=(center_maskcolor[4][2]-5),maskfile[:,:,2]<=(center_maskcolor[4][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[5][2]-5),maskfile[:,:,2]<=(center_maskcolor[5][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[5][2]-5),maskfile[:,:,2]<=(center_maskcolor[5][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[6][2]-5),maskfile[:,:,2]<=(center_maskcolor[6][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[6][2]-5),maskfile[:,:,2]<=(center_maskcolor[6][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[7][2]-5),maskfile[:,:,2]<=(center_maskcolor[7][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[8][2]-5),maskfile[:,:,2]<=(center_maskcolor[8][2]+5))])
    if i==6:
        keepmask0=np.logical_or.reduce([np.logical_and(maskfile[:,:,0]>=(center_maskcolor[10][0]-5),maskfile[:,:,0]<=(center_maskcolor[10][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[11][0]-5),maskfile[:,:,0]<=(center_maskcolor[11][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[12][0]-5),maskfile[:,:,0]<=(center_maskcolor[12][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[13][0]-5),maskfile[:,:,0]<=(center_maskcolor[13][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[14][0]-5),maskfile[:,:,0]<=(center_maskcolor[14][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[15][0]-5),maskfile[:,:,0]<=(center_maskcolor[15][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[16][0]-5),maskfile[:,:,0]<=(center_maskcolor[16][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[17][0]-5),maskfile[:,:,0]<=(center_maskcolor[17][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[18][0]-5),maskfile[:,:,0]<=(center_maskcolor[18][0]+5)),np.logical_and(maskfile[:,:,0]>=(center_maskcolor[19][0]-5),maskfile[:,:,0]<=(center_maskcolor[19][0]+5))])
        keepmask1=np.logical_or.reduce([np.logical_and(maskfile[:,:,1]>=(center_maskcolor[10][1]-5),maskfile[:,:,1]<=(center_maskcolor[10][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[11][1]-5),maskfile[:,:,1]<=(center_maskcolor[11][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[12][1]-5),maskfile[:,:,1]<=(center_maskcolor[12][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[13][1]-5),maskfile[:,:,1]<=(center_maskcolor[13][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[14][1]-5),maskfile[:,:,1]<=(center_maskcolor[14][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[15][1]-5),maskfile[:,:,1]<=(center_maskcolor[15][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[16][1]-5),maskfile[:,:,1]<=(center_maskcolor[16][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[17][1]-5),maskfile[:,:,1]<=(center_maskcolor[17][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[18][1]-5),maskfile[:,:,1]<=(center_maskcolor[18][1]+5)),np.logical_and(maskfile[:,:,1]>=(center_maskcolor[19][1]-5),maskfile[:,:,1]<=(center_maskcolor[19][1]+5))])
        keepmask2=np.logical_or.reduce([np.logical_and(maskfile[:,:,2]>=(center_maskcolor[10][2]-5),maskfile[:,:,2]<=(center_maskcolor[10][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[11][2]-5),maskfile[:,:,2]<=(center_maskcolor[11][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[12][2]-5),maskfile[:,:,2]<=(center_maskcolor[12][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[13][2]-5),maskfile[:,:,2]<=(center_maskcolor[13][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[14][2]-5),maskfile[:,:,2]<=(center_maskcolor[14][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[15][2]-5),maskfile[:,:,2]<=(center_maskcolor[15][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[16][2]-5),maskfile[:,:,2]<=(center_maskcolor[16][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[17][2]-5),maskfile[:,:,2]<=(center_maskcolor[17][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[18][2]-5),maskfile[:,:,2]<=(center_maskcolor[18][2]+5)),np.logical_and(maskfile[:,:,2]>=(center_maskcolor[19][2]-5),maskfile[:,:,2]<=(center_maskcolor[19][2]+5))])
        
    keepmask=np.logical_and(keepmask0,keepmask1)
    keepmask=np.logical_and(keepmask,keepmask2)
    return keepmask



annotation_path='/mnt/annotations/training'
for root, dirs, files in os.walk(annotation_path):
    for file in files:
        save_path=os.path.join('/home/arc-cxj9600/transform-cityspace/annotations_gray256256/training',file.split('.')[0] + '_gray.png')
        maskfile=cv2.imread(os.path.join(annotation_path,file))
        height, width = maskfile.shape[0:2]
        new_height=256
        new_width=256
        maskfile=cv2.resize(maskfile, (new_height, new_width))
        finalmaskfile=np.zeros((new_height,new_width))
        # 在最终的灰度图标签中，0表示背景，1-7分别表示自定义的7个类别
        for i in range(1,8):
            keepmaski=getkeepmask(i)
            finalmaskfile[keepmaski]=i
        cv2.imwrite(save_path,finalmaskfile)
        
