import numpy as np
import cv2

def drawBoundingBoxes(video_image, frame_data, birdv_points, birdv_points_colors, colors, target_size):
    """
    Dibujar a los boundingboxes
    Parámetros:
        video_image: imágen en perspectiva
        frame_data: boundingboxes de las personas en video_image
        birdv_points: lista de puntos en la vista birdview
        birdv_points_colors: puntos organizados segun color
        colors: lista asociada a birdv_points_colors para colorear
        target_size: tamaño final de la imagen
    Salida:
        imagen en perspectiva con boundingboxes
    """    
    red_color = (0,0,255) #BGR
    green_color = (0,255,0)
  
    i = 0
    for index, row in frame_data.iterrows():
        k = birdv_points_colors.index(birdv_points[i])
        color = red_color if colors[k] else green_color
        cv2.rectangle(video_image, ( int(row['bodyLeft']), int(row['bodyTop']) ), ( int(row['bodyRight']), int(row['bodyBottom']) ), color, 2, 1)
        i+=1
    
    return cv2.resize(video_image, target_size)

def getBirdView(points, colors, target_size):
    """
    Generar imagen final birdview
    Parámetros:
        points: lista de puntos en vista birdview organizados
        colors: lista asociada a points para colorear
        target_size: tamaño final de la imagen
    Salida:
        imagen birdview recortada y redimensionada
    """   
    red = (0,0,255) #BGR
    green = (0,255,0)

    # bg de del tamaño original de la birdview
    background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    background[:, :] = [0, 0, 0]

    for i, point in enumerate(points):
        color = red if colors[i] else green
        cv2.circle(background, point, 7, color, -1)

    # area para recortar en la birdview, proporcional a la calle
    cut_posx_min, cut_posx_max = (350, 750)
    cut_posy_min, cut_posy_max = (260, 900)

    bird_eye_view = background[cut_posy_min:cut_posy_max, 
                              cut_posx_min:cut_posx_max, 
                              :]

    return cv2.resize(bird_eye_view, target_size)


def __generate_partial_image(picture, partial_image, position, transparent=False):
    """
    Incrustar imagen parcial en imagen marco.
    Adaptado de: https://github.com/jjrodcast/SocialDistanceDetector/blob/master/utils/view.py
    Parámetros:
        picture: Imagen que va a contener a las vistas parciales, es el marco principal de la aplicación
        partial_image: Imagen parcial que se incrustará en la posición dada
        position: Posición de la imagen parcial
        transparent: Flag para eliminar el background negro
    Salida:
        Imagen marco con la imagen parcial incrustada
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
        

def getIntegratedView(background_image, video_image, birdview_image):
    """
    Generar la imagen final
    Parámetros:
        background_image: imagen de fondo
        video_image: imagen parcial correspondiente a la vista en perspectiva
        birdview_image: imagen parcial correspondiente a la vista bird eye view
    Salida:
        Imagen marco con las imagenes parciales incrustadas
    """
    picture = cv2.imread(background_image)
    content = picture.copy()

    __generate_partial_image(content, video_image, (100, 20))

    __generate_partial_image(content, birdview_image, (100, 752), True)
  
    return content


def pointsColorByDistance(origin_points, min_distance):
    """
    Determinar el incumplimiento de la distancia social a través de los puntos en la birdview
    Parámetros:
        origin_points: puntos en la birdview
        min_distance: distancia minima proporcional
    Salida:
        puntos bird eye re-organizados, lista binaria asociada a los puntos para colorear
    """
    
    colored_points = []
    colors = []

    if len(origin_points) == 0:
        return colored_points, colors
  
    bird_view_points = origin_points[:]

    current_point = bird_view_points[0]
    bird_view_points.remove(current_point)

    current_color = 0
    nearest_color = 0
  
    while len(bird_view_points) > 0:
        
        nearest_distance = None
        nearest_point = None

        for next_point in bird_view_points:
            distance = get_distance(current_point, next_point)
            if(nearest_distance == None or distance < nearest_distance):
                nearest_distance = distance
                nearest_point = next_point

        if (nearest_distance < min_distance):
            current_color = 1
            nearest_color = 1
        elif (nearest_color == 1):
            current_color = 1
            nearest_color = 0
        else:
            current_color = 0
            nearest_color = 0

        colored_points.append(current_point)
        colors.append(current_color)
    
        current_point = nearest_point
        bird_view_points.remove(current_point)
    
    colored_points.append(current_point)
    colors.append(nearest_color)

    return colored_points, colors


def get_distance(x, y):
    '''
    Obtener la distancia euclideana entre dos puntos: x e y
    '''
    x = np.array(x)
    y = np.array(y)
    return np.linalg.norm(x-y)


def framePointsToBirdPoints(frame_points, matrix):
    '''
    Convertir puntos en vista perspectiva a bird eye view points
    Adaptado de: https://stackoverflow.com/questions/45010881/opencv-how-to-use-getperspectivetransform
    Parámetros:
        frame_points: puntos en vista perspectiva
        matrix: matriz de transformación
    '''
    
    #bird_points = []
    #for point in frame_points:
        #homg_point = [point[0], point[1], 1] # homogeneous coords
        #transf_homg_point = matrix.dot(homg_point) # transform
        #transf_homg_point /= transf_homg_point[2] # scale
        #transf_point = transf_homg_point[:2] # remove Cartesian coords
        #bird_points.append((int(transf_point[0]), int(transf_point[1])))

    #points = np.array([[[100, 100]], [[150,100]], [[150,150]], [[150,100]]])
    #homg_points = np.array([[x, y, 1] for [[x, y]] in np.array([frame_points])]).T
    homg_points = np.array([[point[0], point[1], 1] for point in frame_points]).T
    transf_homg_points = matrix.dot(homg_points)
    transf_homg_points /= transf_homg_points[2]
    #transf_points = np.array([[[x,y]] for [x, y] in transf_homg_points[:2].T])
    bird_points = [(int(x),int(y)) for [x, y] in transf_homg_points[:2].T]
    return bird_points

def getFramePoints(frame_data):
    '''
    Se obtiene los puntos base medios de los boundingboxes ('frame_data')
    '''
    return [(row['bodyLeft'] + (row['bodyRight'] - row['bodyLeft'])/2, row['bodyBottom']) for index, row in frame_data.iterrows()]


if __name__ == "__main__":
    print('Funciones utilizadas en el entregable.ipynb')