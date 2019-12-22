# Run this file on server to return a Concentration Index (CI).
# Analysis is in 'Util' folder.

import base64
import io
import sys

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, json, make_response, render_template,
                   request, send_file, send_from_directory)

from flask_cors import CORS, cross_origin
from util.analysis_server import analysis
from PIL import Image
from io import BytesIO
# from StringIO import StringIO

sys.path.append("../")

# from VideoCap import VideoCap


app = Flask(__name__)
cors = CORS(app, resources={r'/*': {"origins": '*'}})
app.config['CORS_HEADER'] = 'Content-Type'
threadDict = {}
ana = analysis()


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))


def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def data_uri_to_cv2_img(uri):
    # encoded_data = uri.split(',')[1]
    nparr = np.fromstring(uri.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.route('/api/v1/predict/', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def predictImage():
    data = request.get_json(force=True)
    # print(data)
    datainString = data.get('image')
    # decoded_data = base64.b64decode(datainString)
    # np_data = np.fromstring(decoded_data,np.uint8)
    img = readb64(datainString)
    cv2.imshow("test", img)
    # print(datainString)
    # img = data_uri_to_cv2_img(data_uri)
    # cv2.imshow(img)
    # image_data = np.asarray(imgdata)
    # print(datainString)
    # print(camData.get('camera_ip_address'))
    # IP = data.get('ipAdd')
    ######## activate with scheduler ########
    # cScheduler = cam_scheduler(cam, schedule_list)
    # cScheduler.start()
    # threadDict[IP] = cScheduler
    # returnData = ana.detect_face(source)

    # cv2.imshow('Frame',source)
    response = make_response('returnData')
    response.headers.set('Content-Type', 'application/json')
    return response


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)
