import cv2
import numpy as np
import glob
import random
import imutils
import functools
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
net = cv2.dnn.readNet(r"C:\Users\HP\Desktop\python\yolotest\numberplate_final.weights", r"C:\Users\HP\Desktop\python\yolotest\numberplate.cfg")
classes=[]
with open(r"C:\Users\HP\Desktop\python\yolotest\classes.names", "r")as f:
    classes=[line.strip() for line in f.readlines()]

images_path=glob.glob(r"C:\Users\HP\Desktop\python\images\f3.jpg")
#h,w=700,400
#images_path=cv2.resize(images_path ,(h,w))

layer_names= net.getLayerNames()
output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors=np.random.uniform(0,255,size=(len(classes),3))

random.shuffle(images_path)
for img_path in images_path:
    img=cv2.imread(img_path)
    img=cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels=img.shape
    
    blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0), True, crop=False)
    net.setInput(blob)
    outs=net.forward(output_layers)

    class_ids=[]
    confidences=[]
    boxes = []
    for out in outs:
        for detection in out:
            scores=detection[5:]
            class_id =np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.3:
                print(class_id)
                center_x =int(detection[0] * width)
                center_y=int(detection[1] * height)
                w=int(detection[2] * width)
                h=int(detection[3] * height)

                x=int(center_x - w / 2)
                y=int(center_y - h / 2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font=cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w,h=boxes[i]
        label= str(classes[class_ids[i]])
        color=colors[class_ids[i]]
        cv2.rectangle(img,(x,y) , (x+w , y+h) , color , 2)
        cv2.putText(img, label,(x-100,y+100),font,3,color,3)
cv2.imshow("images",img)
cv2.waitKey(0)


crop=img[y:y+h, x:x+w]
#h,w=200,50
#crop=cv2.resize(crop,(h,w))

cv2.imshow("image",crop)
cv2.waitKey(0)
gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
filter=cv2.bilateralFilter(gray,11,17,17)

ret, thresh=cv2.threshold(gray,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kern=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
dilation=cv2.dilate(thresh,rect_kern,iterations=1)

cv2.imshow("image",dilation)
cv2.waitKey(0)


custom_config = r'-l nep+eng --psm 6'
a=pytesseract.image_to_string(dilation, config=custom_config)
print(a)
