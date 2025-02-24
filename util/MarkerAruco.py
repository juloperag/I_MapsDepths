import cv2 as cv2
from cv2 import aruco
import numpy as np
import os

#Definiciones generales
ARUCO_DICT = {
    "DICT_4x4_50": aruco.DICT_4X4_50,
    "DICT_4x4_100": aruco.DICT_4X4_100,
    "DICT_4x4_250": aruco.DICT_4X4_250,
    "DICT_4x4_1000": aruco.DICT_4X4_1000,
    "DICT_5x5_50": aruco.DICT_5X5_50,
    "DICT_5x5_100": aruco.DICT_5X5_100,
    "DICT_5x5_250": aruco.DICT_5X5_250,
    "DICT_5x5_1000": aruco.DICT_5X5_1000,
    "DICT_6x6_50": aruco.DICT_6X6_50,
    "DICT_6x6_100": aruco.DICT_6X6_100,
    "DICT_6x6_250": aruco.DICT_6X6_250,
    "DICT_6x6_1000": aruco.DICT_6X6_1000,
    "DICT_7x7_50": aruco.DICT_7X7_50,
    "DICT_7x7_100": aruco.DICT_7X7_100,
    "DICT_7x7_250": aruco.DICT_7X7_250,
    "DICT_7x7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11, 
}

def generate_markers(marker_size, type_aruco, marker_id):
    """Generador de codigo Aruco 
    Input:
        marker_size: Tamaño Marcador en pixeles
        type_aruco: Tipo marcador Aruco, ver Aruco_dict
        id: Identificador id del Marcador Aruco
    """    
    # Crear carpeta para guardar los marcadores si no existe
    os.makedirs("./markers", exist_ok=True)
    # Especificar tipo de diccionario ArUco
    marker_dict = aruco.getPredefinedDictionary(type_aruco)
    # Generar imagen del marcador ArUco
    marker_image = aruco.generateImageMarker(marker_dict, marker_id, marker_size)
    # Visualizar el marcador
    cv2.imshow("Aruco Marker", marker_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar el marcador
    filename = f"./markers/marker_{marker_id}.png"
    cv2.imwrite(filename, marker_image)
    print(f"Marcador guardado en {filename}")


def geometry_estimation(image, side, type_aruco, camera_matrix, dis_coeffs):
    """Estimacion de la distancia y angulo del marcador Aruco
    Input:
        image: Imagen de entrada
        side: Tamaño de uno de los lados del marcador en mm
        type_aruco: Tipo marcador Aruco, ver Aruco_dict
        camera_matrix: Matriz intrinsica de la camara
        dis_coeffs: coeficientes de dispersion distorsión
    Output:
        Diccionario con los elementos distance, theta_XY
    """
    #Conversion imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray_image is None:
        print("Error: La imagen no se cargó correctamente.")
    #Definicion parametros y diccionario
    marker_dict = aruco.getPredefinedDictionary(type_aruco)
    param_markers = aruco.DetectorParameters()
    #Deteccion marcador Aruco
    aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)
    marker_corners, marker_ids, reject_img_point = aruco_detector.detectMarkers(gray_image)
    #Si se detecto un marcador se estima la pose del mismo
    if marker_corners:
        #Construccion vertices 3D
        img_points = marker_corners[0].reshape(4, 2)
        corner_points = np.array([
            [0,  side,  0],  # Vértice superior izquierdo (0)
            [side, side, 0],  # Vértice superior derecho (1)
            [side, 0,  0],  # Vértice inferior derecho (2)
            [0,  0,  0]   # Vértice inferior izquierdo (3)
        ], dtype=np.float32)
        obj_points = corner_points - corner_points[0] 
        # Resolver PnP con respecto al vértice seleccionado con vertex_refence
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dis_coeffs)
        R, _ = cv2.Rodrigues(rvec)
        #Calculo distancia
        R, _ = cv2.Rodrigues(rvec)
        vertex_cam = R @ obj_points[0].reshape(3, 1) + tvec
        distance = np.linalg.norm(vertex_cam)
        # Calculo angulo entre eje Z de la camara y eje Z marcador Aruco
        z_marker = R[:, 2]  
        z_camera = np.array([0, 0, 1])
        cos_theta = np.dot(z_marker, z_camera) / (np.linalg.norm(z_marker) * np.linalg.norm(z_camera))
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Asegurar valores en el rango válido
        #Definicion de la posicion de las esquinas del marcador
        parameter_geometry = {"distance": distance, "theta_XY":  angle_rad, "normal": z_marker}
        #Retornar
        return parameter_geometry
    else:
        print("No se identifico ningun marcador Aruco")

def location_corners(image, type_aruco):
    """Localizacion de las esquinas del marcador Aruco
    Input:
        image: Imagen de entrada
        type_aruco: Tipo marcador Aruco, ver Aruco_dict
    Output:
        Diccionario con los elementos top left, top right, bottom right, bottom left
    """
    #Conversion imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray_image is None:
        print("Error: La imagen no se cargó correctamente.")
    #Definicion parametros y diccionario
    marker_dict = aruco.getPredefinedDictionary(type_aruco)
    param_markers = aruco.DetectorParameters()
    #Deteccion marcador Aruco
    aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)
    marker_corners, marker_ids, reject_img_point = aruco_detector.detectMarkers(gray_image)
    #Si se detecto un marcador se estima la pose del mismo
    if marker_corners:
        #Construccion vertices 3D
        img_points = marker_corners[0].reshape(4, 2)
        parameter_corners = {"top_left": img_points[0].ravel(), "top_right": img_points[1].ravel(), 
                             "bottom_right": img_points[2].ravel(), "bottom_left": img_points[3].ravel()}
        #Retornar
        return parameter_corners
    else:
        print("No se identifico ningun marcador Aruco")


def draw_parameter_detection(image, parameter_geometry, parameter_corners):
    """Indicador de las esquinas y distancia del marcador detectado
    Input:
        image: Imagen de entrada
        parameter_detection: Diccionario con los elementos distance, vertex reference, top left, top right, bottom right, bottom left
    Output:
        Imagen con los indicadores detectados 
    """
    image_copy = np.copy(image)
    #Dibujo esquinas
    list_corners = ["top_left","top_right", "bottom_right", "bottom_left"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # Rojo, Verde, Azul, Amarillo
    labels = ["0", "1", "2", "3"]
    for i in range(0, len(list_corners)):
        x, y = int(parameter_corners[list_corners[i]][0]), int(parameter_corners[list_corners[i]][1])
        cv2.circle(image_copy, (x, y), 30, colors[i], -1)  # Dibujar punto
        cv2.putText(image_copy, labels[i], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 7, colors[i], 2)

    #Se pone el valor de la distancia sobre el vertice respectivo
    distance_value = parameter_geometry["distance"]
    coord = (int(parameter_corners["top_left"][0]) , int(parameter_corners["top_left"][1]))
    cv2.putText(
        image_copy,
        f"Dist: {round(distance_value, 2)}",
        coord,
        cv2.FONT_HERSHEY_PLAIN, 15, (0, 0, 255), 2,cv2.LINE_AA,
    )
    #Retornar valor
    return image_copy