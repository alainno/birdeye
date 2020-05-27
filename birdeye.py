import cv2
import numpy as np
from functions import getBirdView

vs = cv2.VideoCapture('TownCentreXVID.avi')

w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)*0.3)
h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.3)

while True:
  
  (grab, frame) = vs.read()

  if not grab:
    break

  frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

  #frame = cv2.resize(frame, (w+200,h))
  #print(frame.shape)
  #a = np.random.rand(N,N)
  #print(type(frame2))
  frame2 = getBirdView([(2100,300),(2200,350)], [1,0], (200,h))
  
  frame3 = np.concatenate((frame, frame2), axis=1)
  #print(frame.shape)

  #integrated_view = getIntegratedView()

  cv2.imshow("Tracking", frame3)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

vs.release()
cv2.destroyAllWindows()