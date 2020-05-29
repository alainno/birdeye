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

def __generate_partial_image(picture, partial_image, position, transparent=False):
    """
    Esta función genera incrusta una imagen parcial a nuestro picture final
    Parámetros:
        picture: Imagen que va a contener a las vistas parciales, es el marco principal de la aplicación. 
                 Debe generarse una única vez, recomendable usa las dimensiones  (1250, 2600, 3). 
                 Usar la función 'generate_picture' para obtener este parámetro.
        partial_image: Imagen parcial que se incrustará en una posición dada dentro del picture
        position: Posición de la imagen parcial
    Salida:
        Retorna la imagen del picture con la vista parcial incrustada
    """
    if not isinstance(position, tuple):
        raise Exception("position must be a tuple representing x,y coordinates")
    
    image_height, image_width = partial_image.shape[:2]
    x, y = position

    if transparent:
      foreground = partial_image[:,:]>0
      #print("foreground")
      #print(foreground)
      #picture[x: x + image_height, y: y + image_width] = partial_image[foreground]
      picture[x: x + image_height, y: y + image_width][foreground] = partial_image[foreground]
    else:
      picture[x: x + image_height, y: y + image_width] = partial_image

def getIntegratedView(video_image, birdview_image):
  picture = cv2.imread('frameview.png')
  content = picture.copy()

  __generate_partial_image(content, video_image, (100, 20))

  __generate_partial_image(content, birdview_image, (100, 752), True)
  
  return content

if __name__ == "__main__":
    bird_view = getBirdView([(2100,300)], [1], (200,400))
    cv2.imshow("hola", cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()