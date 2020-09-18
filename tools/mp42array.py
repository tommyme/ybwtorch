
# 获取某视频的第50帧
import cv2
 
cap = cv2.VideoCapture('XXX.avi')  #返回一个capture对象
cap.set(cv2.CAP_PROP_POS_FRAMES,50)  #设置要获取的帧号
a,b=cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
cv2.imshow('b', b)
cv2.waitKey(1000)


# 将整个视频读取成一个四阶的numpy array，即它的shape为(帧数，高，宽，通道数3)

import cv2
 
cap = cv2.VideoCapture('XXX.mp4')
wid = int(cap.get(3))
hei = int(cap.get(4))
framerate = int(cap.get(5))
framenum = int(cap.get(7))
 
video = np.zeros((framenum,hei,wid,3),dtype='float16')
cnt = 0
while(cap.isOpened()):
    a,b=cap.read()
    cv2.imshow('%d'%cnt, b)
    cv2.waitKey(20)
    b = b.astype('float16')/255
    video[cnt]=b
    print(cnt)
    cnt+=1
