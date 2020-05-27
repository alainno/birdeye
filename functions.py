import numpy as np
import cv2

def getBirdView(points, colors, target_size):

  red = (255,0,0)
  green = (0,255,0)
  #target_size = (200, 500)

  background = np.zeros((3000, 4500, 3), dtype=np.uint8)
  background[:, :] = [0, 0, 0]


  for i, point in enumerate(points):
    color = red if colors[i] else green
    print('point:', point)
    print('color:', color)
    cv2.circle(background, point, 25, color, -1)

  # ROI of bird eye view
  cut_posx_min, cut_posx_max = (2000, 3400)
  cut_posy_min, cut_posy_max = ( 200, 2800)

  bird_eye_view = background[cut_posy_min:cut_posy_max, 
                              cut_posx_min:cut_posx_max, 
                              :]

  # Bird Eye View resize
  return cv2.resize(bird_eye_view, target_size)

  #frame2 = np.ones((h,200,3),np.uint8)
  #frame2[:,:]=(255,0,0)
  #return frame2

if __name__ == "__main__":
    bird_view = getBirdView([(2100,300)], [1], (200,400))
    cv2.imshow("hola", cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()