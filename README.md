# Distanciamento Social
### :pencil2: Tarea 1 de Computación Gráfica - INF658
Este programa analiza un video captado por una cámara en una calle e identifica a las personas que no cumplen con el distanciamiento social, siendo útil como apoyo para la disminución de contagios del nuevo coronavirus COVID-19.

***

**Realizado por:** [Alain M. Alejo Huarachi](//dealain.com)

**Profesor:** [Dr. Iván Sipirán Mendoza](//ivan-sipiran.com)

***

### :video_camera: Resultado

[![RESULTADO](https://img.youtube.com/vi/4grqapP_r6I/0.jpg)](https://www.youtube.com/watch?v=PPMXI7OtAig)

### Procedimiento

1. Selección del rectángulo a proyectar, en la vista de cámara.
2. Transformar la vista de cámara (perspectiva) a una vista aérea plana (bird eye view).
3. Selección de puntos de referencia de los cuadros delimitadores de las personas (bounding boxes).
4. Proyección de los puntos de referencia desde la vista en perspectiva hacia la bird eye view.
5. Determinación de los puntos que no cumplen con la distancia mínima.
6. Implementación del video de seguimiento.

### :page_facing_up: Requerimientos

* Python, Open CV, Numpy, Pandas, Myplotlib.
* Entornos recomendados:
    - JupyterLab.
    - Google Colaboratory.

### :scroll: Scripts

* Notebook con la ejecución del procedimiento paso a paso: [entregable.ipynb](/entregable.ipynb).
* Funciones de apoyo: [functions.py](/functions.py).

### :blue_book: Referencias

* [Landing AI Creates an AI Tool to Help Customers Monitor Social Distancing in the Workplace](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/).
* [Social Distancing Detector](https://github.com/jjrodcast/SocialDistanceDetector).






