
import cv2
import os
 
 
def images_to_video():
    fps = 5  # 帧率
    num_frames = 500
    img_array = []
    img_width = 1344
    img_height = 756
    path = "E:/dataset/video/3/"
    for file_name in os.listdir(path):
        img = cv2.imread(path + file_name)
        if img is None:
            print(file_name + " is non-existent!")
            continue
        img_array.append(img)
 
    # out = cv2.VideoWriter('demo_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (img_width, img_height))
    out = cv2.VideoWriter('demo_people.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_width, img_height))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
 
 
def main():
    images_to_video()