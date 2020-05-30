import numpy as np
import cv2

def drawBoundingBoxes(video_image, frame_data):
  for index, row in frame_data.iterrows():
    cv2.rectangle(video_image, ( int(row['bodyLeft']), int(row['bodyTop']) ), ( int(row['bodyRight']), int(row['bodyBottom']) ), (0,255,0), 2, 1)
  return video_image

def getBirdView(points, colors, target_size):

  red = (0,0,255) #BGR
  green = (0,255,0)

  background = np.zeros((3000, 4500, 3), dtype=np.uint8)
  background[:, :] = [0, 0, 0]


  for i, point in enumerate(points):
    color = red if colors[i] else green
    #print('point:', point)
    #print('color:', color)
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
      picture[x: x + image_height, y: y + image_width][foreground] = partial_image[foreground]
    else:
      picture[x: x + image_height, y: y + image_width] = partial_image

def getIntegratedView(video_image, birdview_image):
  picture = cv2.imread('frameview.png')
  content = picture.copy()

  __generate_partial_image(content, video_image, (100, 20))

  __generate_partial_image(content, birdview_image, (100, 752), True)
  
  return content

def pointsColorByDistance(origin_points, min_distance):
  
  colored_points = []
  colors = []

  if len(origin_points) == 0:
    return colored_points, colors
  
  bird_view_points = origin_points[:]

  current_point = bird_view_points[0]
  #print('current point: ', current_point)

  bird_view_points.remove(current_point)

  current_color = 0
  nearest_color = 0
  
  while len(bird_view_points) > 0:
    
    #print('Current point:', current_point)

    nearest_distance = None
    nearest_point = None

    for next_point in bird_view_points:

      #print('next point: ', next_point)

      distance = get_distance(current_point, next_point)

      #print('distance: ', distance)

      if(nearest_distance == None or distance < nearest_distance):
        nearest_distance = distance
        nearest_point = next_point

    #print('Nearest point:', nearest_point)

    if (nearest_distance < min_distance):
      #print('color: red')
      current_color = 1
      nearest_color = 1
    elif (nearest_color == 1):
      #print('color: red and green')
      current_color = 1
      nearest_color = 0
    else:
      #print('color: green')
      current_color = 0
      nearest_color = 0

    colored_points.append(current_point)
    colors.append(current_color)

    #print(current_point, ' => ', current_color)
    
    current_point = nearest_point
    bird_view_points.remove(current_point)

  #colored_points.append((current_point[0],current_point[1],next_color))
  colored_points.append(current_point)
  colors.append(nearest_color)

  #print(current_point, ' => ', nearest_color)

  return colored_points, colors

def get_distance(x, y):
  x = np.array(x)
  y = np.array(y)
  return np.linalg.norm(x-y)

def framePointsToBirdPoints(frame_points, matrix):
  bird_points = []
  for point in frame_points:
    homg_point = [point[0], point[1], 1] # homogeneous coords
    transf_homg_point = matrix.dot(homg_point) # transform
    transf_homg_point /= transf_homg_point[2] # scale
    transf_point = transf_homg_point[:2] # remove Cartesian coords
    bird_points.append((int(transf_point[0]), int(transf_point[1])))
  return bird_points

def getFramePoints(frame_data):
  return [(row['bodyLeft'] + (row['bodyRight'] - row['bodyLeft'])/2, row['bodyBottom']) for index, row in frame_data.iterrows()]

def getMatrix(birdeye_width, birdeye_height):
  #birdeye_width = 900
  #birdeye_height = 600

  #Consideramos una imagen final de tamaño:
  dst_size=(birdeye_width,birdeye_height)

  #Obtenemos los puntos de referencia de la imagen original:
  src=np.float32([[1187, 178], [1575, 220], [933,883], [295, 736]])

  #Obtenemos los puntos de destino, como una proporción de la imagen de destino:
  dst=np.float32([(0.57,0.42), (0.65, 0.42), (0.65,0.84), (0.57,0.84)])

  #Obtenemos los puntos de destino al multiplicar la proporción por el tamaño de la imagen:
  dst = dst * np.float32(dst_size)

  #Calculamos la matriz de homografía para la transformación:
  return cv2.getPerspectiveTransform(src, dst)  

if __name__ == "__main__":
    bird_view = getBirdView([(2100,300)], [1], (200,400))
    cv2.imshow("hola", cv2.cvtColor(bird_view, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #cv2.destroyAllWindows()