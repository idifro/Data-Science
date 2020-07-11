from flask import Response,request,render_template
import jsonpickle
from flask import Flask
import numpy as np
import cv2
from keras.models import model_from_json
import os
import json

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded model from disk")


prototxtPath = 'Face Detection Cafe Model/deploy.prototxt'
weightsPath = 'Face Detection Cafe Model/res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNet(prototxtPath,weightsPath)



addr = 'http://127.0.0.2:5000'
test_url = addr + '/api/test'
content_type = 'image/jpeg'
headers = {'content_type': content_type}




app = Flask(__name__)

@app.route("/")
def home():
	return render_template("index.html")


@app.route('/api/test/', methods=['POST'])
def test():
	r = request
	nparr = np.fromstring(r.data,np.uint8)
	img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
	(h,w) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	value = []
	co_or = []
	for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
		confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
		if confidence > 0.2:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the dimensions of
            # the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			co_or.append([startX,startY,endX,endY])
			img_crp = img[startY:endY,startX:endX]
			img_crp = cv2.cvtColor(img_crp,cv2.COLOR_BGR2GRAY)
			img_crp = cv2.resize(img_crp,(50,50))
			img_crp = img_crp.reshape(-1,50,50,1)
			img_crp = img_crp/255
            ##print(img_crp.shape)
			pred = loaded_model.predict(img_crp.reshape(-1,50,50,1))
			my_list = map(lambda x: x[0], pred)
			pred = list(my_list)[0]
			if pred > 0.3:
				value.append(1)
			else:
				value.append(0)


	response = {'Values':str(value),'Boxes':str(co_or)}

	response_pickled = jsonpickle.encode(response)
	return Response(response=response_pickled, status=200, mimetype="application/json")



if __name__ == '__main__':
   app.run(host = '127.0.0.2' ,port = 5000)
