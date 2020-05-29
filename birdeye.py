import cv2
import numpy as np
#from functions import getBirdView, getIntegratedView, pointsColorByDistance, framePointsToBirdPoints, getFramePoints
from functions import *
import pandas as pd

data = pd.read_csv('data/TownCentre-groundtruth.top.txt')
data.columns = ['numPersona','numFrame','headValid','bodyValid','headLeft','headTop','headRight','headBottom','bodyLeft','bodyTop','bodyRight','bodyBottom']

vs = cv2.VideoCapture('data/TownCentreXVID.avi')

w = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)

matrix = getMatrix(4500, 3000)

#h = 400
#w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)/vs.get(cv2.CAP_PROP_FRAME_HEIGHT))*h

#quit()

i=0
stop_frame=500

while (True and i<stop_frame):
  
  (grab, frame) = vs.read()

  if not grab:
    break

  fy = 400/h
  fx = fy
  video_image = cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

  #frame_points = [()]

  frame_data = data[data['numFrame']==i]

  frame_points = getFramePoints(frame_data)

  points = framePointsToBirdPoints(frame_points, matrix)
  #points = [(2100,300),(2200,350)]
  #colors = [1,0]

  birdv_points, colors = pointsColorByDistance(points, 100)

  birdv_image = getBirdView(birdv_points, colors, (250,400))

  integrated_image = getIntegratedView(video_image, birdv_image)

  cv2.imshow("Distanciamiento Social", integrated_image)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  i+=1

vs.release()
cv2.destroyAllWindows()