import cv2

image = cv2.imread('images/trump.jpg', 0)

# find faces:
cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt2.xml")
faces = cascade.detectMultiScale(image, 1.3, 5)

# create landmark detector and load lbf model:
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("models/lbfmodel.yaml")

# run landmark detector:
ok, landmarks = facemark.fit(image, faces)

# print results:
print("landmarks LBF",ok, landmarks)