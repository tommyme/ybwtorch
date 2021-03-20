import cv2


def draw(xywhLabels, img_path, color):
    font = cv2.FONT_HERSHEY_COMPLEX
    img = cv2.imread(img_path)
    for xywhLabel in xywhLabels:
        x, y, w, h, label = xywh
        # color (0,255,0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
        cv2.putText(img, label, (x-10, y-10), font,
                    0.5, (255, 0, 255), thickness=1)
    return img


[[1, 2, 3, 4, 'ybw'], [1, 2, 3, 4, '']]
# x = 119
# y = 111
# w = 163-119
# h = 150-111

# draw(x, y, w, h, '7.jpg', (255, 0, 255), 'red', 0.98)
