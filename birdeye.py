import cv2
import numpy as np
from functions import getBirdView, getIntegratedView

vs = cv2.VideoCapture('TownCentreXVID.avi')

w = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)


#h = 400
#w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)/vs.get(cv2.CAP_PROP_FRAME_HEIGHT))*h

#quit()

i=0

while (True and i<100):
  
  (grab, frame) = vs.read()

  if not grab:
    break

  #fy = 400/h
  #fx = (w/h)*fy
  fy = 400/h
  frame = cv2.resize(frame, None, fx=fy, fy=fy, interpolation=cv2.INTER_CUBIC)

  #frame = cv2.resize(frame, (w+200,h))
  #print(frame.shape)
  #a = np.random.rand(N,N)
  #print(type(frame2))

  points = [(2100,300),(2200,350)]
  colors = [1,0]

  frame2 = getBirdView(points, colors, (200,400))

  integrated_image = getIntegratedView(frame, frame2)

  cv2.imshow("Distanciamiento Social", integrated_image)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  i+=1

vs.release()
cv2.destroyAllWindows()