import torch
import cv2
import MLPModel.Model_2 as Model_2
model = Model_2.MLP(9216,30).double()
model.load_state_dict(torch.load('../MLPModel/Model2Model.pth'))
img = cv2.imread('face_ex.jpg')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(96, 96)
)

for (x, y, w, h) in face:
    scaled_face=cv2.resize(gray_image[y:y+h, x:x+w],(96,96),interpolation=cv2.INTER_LINEAR)
    inp=torch.tensor(scaled_face.flatten(),dtype=torch.float64)/255
    print(w,h)
    with torch.no_grad():
        res=model(inp)
    for idx in range(0,30,2):
        feat_x,feat_y=res[idx].item(), res[idx+1].item()
        feat_x = (feat_x+1)*w/2
        feat_y = (feat_y+1)*h/2
        middle = (int(feat_x+x),int(feat_y+y))
        cv2.circle(img, middle, 5, color=(0, 0, 255),thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('testS', img)
    cv2.waitKey()
