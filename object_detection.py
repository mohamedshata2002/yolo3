import cv2 as cv
import os 
import numpy as np 
### loading the model
os.chdir('D:\WorkStation\object\yolo\yolo3')
model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

with open ('coco.names','r') as file:
    classes = [line.strip() for line in file.readlines()]
    #print(classes)
    print(len(classes))
    
    
layers = model.getLayerNames()
print(len(layers))

layer = model.getUnconnectedOutLayers()

out_layer = [layers[i-1] for i in model.getUnconnectedOutLayers()]
print(out_layer)

colors = np.random.uniform(0,255,size=(len(classes),3))
### video preprocessing
video = cv.VideoCapture('people.mp4')
fourcc = cv.VideoWriter_fourcc('X','V','I','D')
out_v = cv.VideoWriter('out.avi',fourcc,20,(1280,720))
while True:
    ret,frame = video.read()
    if ret is not True:
        break
    frame =cv.resize(frame,None,fx=0.4,fy=0.4)
    blob =cv.dnn.blobFromImage(frame,1/255,(416,416),(0,0,0),crop=False)
    model.setInput(blob)
    outs = model.forward(out_layer)
    ### preprocessing the output
    bndb = []
    confidences = []
    out_classes = []
    height,width ,channel = frame.shape
    for out in outs :
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>0.2:
                x_mid = int(detection[0]*width)
                y_mid = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(x_mid-(w/2))
                y = int(y_mid-(h/2))
                confidences.append(float(confidence))
                out_classes.append(class_id)
                bndb.append([x,y,w,h])
    index=cv.dnn.NMSBoxes(bndb, confidences, 0.5, 0.4)
    font = cv.FONT_HERSHEY_SIMPLEX
    for i in range (len(bndb)):
        if i in index:
            x,y,w,h = bndb[i]
            label = str(classes[out_classes[i]])
            color = colors[out_classes[i]]
            cv.rectangle(frame,(x,y),(x+w,y+h) , color,2)
            cv.putText(frame, label, (x,y-10), font,0.5,color,1)
    frame = cv.resize(frame, (1280,720))
    out_v.write(frame)
    cv.imshow('window',frame)
    key = cv.waitKey(30)
    if key==27:
        break
cv.destroyAllWindows()
    
    

