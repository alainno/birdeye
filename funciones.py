import numpy as np
import cv2

def drawBoundingBoxes(video_image, frame_data, birdv_points, colors, lines, target_size):
    """
    Dibujar a los boundingboxes
    Parámetros:
        video_image: imágen en perspectiva
        frame_data: boundingboxes de las personas en video_image
        birdv_points: lista de puntos en la vista birdview
        colors: lista asociada a birdv_points_colors para colorear
        lines: lineas entre distancias incumplidas
        target_size: tamaño final de la imagen
    Salida:
        imagen en perspectiva con boundingboxes
    """    
    red_color = (0,0,255) #BGR
    green_color = (0,255,0)
  
    i = 0
    for index, row in frame_data.iterrows():
        color = red_color if colors[i] else green_color
        cv2.rectangle(video_image, ( int(row['bodyLeft']), int(row['bodyTop']) ), ( int(row['bodyRight']), int(row['bodyBottom']) ), color, 2, 1)
        i+=1
    
    for line in lines:
        a = birdv_points.index(line[0])
        b = birdv_points.index(line[1])
        
        row_a = frame_data.iloc[a]
        row_b = frame_data.iloc[b]

        point_a = getFramePoint(row_a)
        point_b = getFramePoint(row_b)
        
        cv2.line(video_image, point_a, point_b, red_color, thickness=2)
    
    return cv2.resize(video_image, target_size)


def getFramePoint(row):
    '''
    Obtiene el punto medio en la base del boundingbox
    '''
    return (int(row['bodyLeft'] + (row['bodyRight'] - row['bodyLeft'])/2), int(row['bodyBottom']))


def getBirdView(points, colors, lines, target_size):
    """
    Generar imagen final birdview
    Parámetros:
        points: lista de puntos en vista birdview organizados
        colors: lista asociada a points para colorear
        lines: lineas entre distancias incumplidas
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
    
    for line in lines:
        cv2.line(background, line[0], line[1], red, thickness=2)

    # area para recortar en la birdview, proporcional a la calle
    cut_posx_min, cut_posx_max = (350, 750)
    cut_posy_min, cut_posy_max = (250, 890)

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
        

def getIntegratedView(background_image, video_image, birdview_image, percent):
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
    
    cv2.putText(content, "Distancia: "+str(percent)+"%", (25,115), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    cv2.putText(content, "No cumple: "+str(100-percent)+"%", (25,130), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
  
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

    return colored_points, colors, int((colors.count(0)/len(colors))*100)


def pointsColorByDistanceLines(origin_points, min_distance):
    """
    Determinar el incumplimiento de la distancia social a través de los puntos en la birdview
    Parámetros:
        origin_points: puntos en la birdview
        min_distance: distancia minima proporcional
    Salida:
        puntos bird eye re-organizados, lista binaria asociada a los puntos para colorear
    """
    bird_view_points = origin_points[:]
    n = len(origin_points)
    colors = [0] * n
    lines = []
    
    for i,current_point in enumerate(origin_points):
        bird_view_points.remove(current_point)
        for j,next_point in enumerate(bird_view_points):
            distance = get_distance(current_point, next_point)
            if distance < min_distance:
                colors[i] = 1
                colors[i+j+1] = 1
                lines.append((current_point,next_point))
    
    percent = 0 if n==0 else int((colors.count(0)/len(colors))*100)
    
    return colors, lines, percent


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
    homg_points = np.array([[point[0], point[1], 1] for point in frame_points]).T
    transf_homg_points = matrix.dot(homg_points)
    transf_homg_points /= transf_homg_points[2]
    bird_points = [(int(x),int(y)) for [x, y] in transf_homg_points[:2].T]
    return bird_points

def getFramePoints(frame_data):
    '''
    Se obtiene los puntos base medios de los boundingboxes ('frame_data')
    '''
    return [(row['bodyLeft'] + (row['bodyRight'] - row['bodyLeft'])/2, row['bodyBottom']) for index, row in frame_data.iterrows()]


if __name__ == "__main__":
    print('Funciones utilizadas en el entregable.ipynb')