import numpy as np
import cv2
import open3d as o3d
from scipy.ndimage import binary_dilation
from scipy.spatial import Delaunay
import math
import trimesh

def lines_to_mark_segmentation(lines, class_selection, height, width):
    """Funcion para convertir lineas referentes un archivo file que contiene la segmentacion de 
       los objetos en YOLO a una mascara.
    Input:
        lines: lista de las lineas luego de leer el archivo txt
        class_selection: Clase especifica en la mascara
        height, width: Dimensiones de la imagen
    Output:
        Mascara de segmentacion
    """""
    #Creamos matriz para definir la mascara 
    mark_segmentation = np.zeros((height, width))
    #Procesar las lineas del archivo de segmentacion
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        #Deacuerdo a la clase se dibuja la segmentacion
        if(class_id == class_selection):
            points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)  # Reshape into pairs (x, y)       
            # Denormalize coordinates
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(int)
            # Draw polygon on the image
            cv2.polylines(mark_segmentation, [points], isClosed=True, color=(255, 255, 255), thickness=2)
            cv2.fillPoly(mark_segmentation, [points], color=(255, 255, 255, 200))  # Semi-transparent fill
            #Convertir archivo en binario
            mark_segmentation = mark_segmentation.astype(bool)
    return mark_segmentation 


def calculation_volumen(map_depth, mask, normal, cmtrix):
    """Funcion para calculo de volumen 
    Input:
        map_depth: Mapa de profundidad
        mask: Mascara del objeto para su reconstruccion 3d
        normal: Normal inversa al plano de la mesa
        cmtrix: Matriz intrinseca de la camara 
    Output:
        Volumen calculado en cm3
    """""
    #Busqueda valor D
    #Definicion de geometria
    width, height = int(np.shape(map_depth)[1]), int(np.shape(map_depth)[0])
    v, u = np.where(mask)
    # Reconstruir coordenadas 3D
    Z = map_depth[v,u]
    # Convertir coordenadas de píxeles a coordenadas de cámara
    xx = (u - width/2) / cmtrix[0,0]
    yy = (v - height/2) / cmtrix[1,1]
    # Multiplicar por la profundidad para obtener puntos en 3D
    X = np.multiply(xx, Z)
    Y = np.multiply(yy, Z)
    # Creacion nube de puntos
    points3d_segmented = np.vstack((X, Y, Z)).T
    #Busqueda de punto mas alejado en la direccion de la normal inversa
    proy= np.dot(points3d_segmented, normal)
    indice_max = np.argmax(proy)
    point_off = points3d_segmented[indice_max]
    #Definicion de variables a partir de parametros
    D =  - np.dot(normal, point_off)
    #Mapa de profundidad a nube de puntos cerrada 
    pcd = depth_to_point_cloud_segmented(map_depth, mask, normal, cmtrix)
    #Transformacion de coordenadas 
    [a, b, c] = normal
    d = D
    pcd_modification = pcd.translate((0,0,d/c))
    cos_theta = c / math.sqrt(a**2 + b**2 + c**2)
    sin_theta = math.sqrt((a**2+b**2)/(a**2 + b**2 + c**2))
    u_1 = b / math.sqrt(a**2 + b**2)
    u_2 = -a / math.sqrt(a**2 + b**2)
    rotation_matrix = np.array([[cos_theta + u_1**2 * (1-cos_theta), u_1*u_2*(1-cos_theta), u_2*sin_theta],
                                [u_1*u_2*(1-cos_theta), cos_theta + u_2**2*(1- cos_theta), -u_1*sin_theta],
                                [-u_2*sin_theta, u_1*sin_theta, cos_theta]])
    pcd_modification.rotate(rotation_matrix)
    #Triangulacion de nube de puntos 
    downpdc = pcd_modification.voxel_down_sample(voxel_size=0.05)
    xyz = np.asarray(downpdc.points)
    xy_catalog = xyz[:, :2]
    tri = Delaunay(xy_catalog)
    #Creacion de superfice
    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(xyz)
    surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
    # Calcular volumen total (cm³)
    mesh = trimesh.Trimesh(vertices=surface.vertices, faces=surface.triangles)
    volume = abs(mesh.volume)/1e3
    #Retornar valor
    return volume

def depth_to_point_cloud_scene(map_depth, img_color, cmtrix):
    """Funcion para constuir una nube de puntos de la escena completa
    Input:
        map_depth: Mapa de profundidad
        img_color: Imagen a color 
        cmtrix: Matriz intrinseca de la camara 
    Output:
        Nube de puntos de la imagen completa
    """""
    #Creacion de geometria
    width_dst, height_dst = int(np.shape(img_color)[1]), int(np.shape(img_color)[0])
    u, v = np.meshgrid(np.arange(width_dst), np.arange(height_dst))
    # Reconstruir coordenadas 3D
    Z = map_depth
    # Convertir coordenadas de píxeles a coordenadas de cámara
    x = (u - width_dst/2) / cmtrix[0,0]
    y = (v - height_dst/2) / cmtrix[1,1]
    # Multiplicar por la profundidad para obtener puntos en 3D
    X = np.multiply(x, Z)
    Y = np.multiply(y, Z)
    # Crear la nube de puntos
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = np.array(img_color).reshape(-1, 3) / 255.0
    # Crear objeto PointCloud de Open3D
    points_cloud = o3d.geometry.PointCloud()
    points_cloud.points = o3d.utility.Vector3dVector(points)
    points_cloud.colors = o3d.utility.Vector3dVector(colors)
    #Retornar nube de puntos
    return points_cloud


def depth_to_point_cloud_segmented(map_depth, mask, normal, cmtrix):
    """Funcion para constuir una nube de puntos de un objeto segmentado en la escena
    Input:
        map_depth: Mapa de profundidad
        img_color: Imagen a color 
        mask: Mascara del objeto para su reconstruccion 3d
        normal: Normal inversa al plano de la mesa
        cmtrix: Matriz intrinseca de la camara 
    Output:
        Nube de puntos de la segmentacion
    """""
    #####-----------------------Nube de puntos segmentada----------------------#######
    #Definicion de geometria
    width, height = int(np.shape(map_depth)[1]), int(np.shape(map_depth)[0])
    v, u = np.where(mask)
    # Reconstruir coordenadas 3D
    Z = map_depth[v,u]
    # Convertir coordenadas de píxeles a coordenadas de cámara
    x = (u - width/2) / cmtrix[0,0]
    y = (v - height/2) / cmtrix[1,1]
    # Multiplicar por la profundidad para obtener puntos en 3D
    X = np.multiply(x, Z)
    Y = np.multiply(y, Z)
    # lista de nube de puntos
    points3d_segmented = np.vstack((X, Y, Z)).T
    #####----------------------Nube de puntos plano----------------------#######
    #Busqueda de punto mas alejado en la direccion de la normal inversa
    proy= np.dot(points3d_segmented, normal)
    indice_max = np.argmax(proy)
    point_off = points3d_segmented[indice_max]
    #Definicion de variables a partir de parametros
    D = - np.dot(normal, point_off)
    #Proyeccion de puntos sobre plano de la mesa
    lambdas = (np.dot(points3d_segmented, normal) + D) 
    lambdas = lambdas[:, np.newaxis]
    points3d_plane = points3d_segmented - lambdas * normal
    #####----------------------Nube de puntos lateral----------------------#######
    #Hallar borde de la mascara
    border = (binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)))^mask  # XOR lógico
    vb, ub = np.where(border)
    # Reconstruir coordenadas 3D
    Zb = map_depth[vb,ub]
    # Convertir coordenadas de píxeles a coordenadas de cámara
    xb = (ub - width/2) / cmtrix[0,0]
    yb = (vb - height/2) / cmtrix[1,1]
    # Multiplicar por la profundidad para obtener puntos en 3D
    Xb = np.multiply(xb, Zb)
    Yb = np.multiply(yb, Zb)
    # lista de nube de puntos
    points3d_borde = np.vstack((Xb, Yb, Zb)).T
    #Definicion de lista de puntos y distancia estre puntos
    list_points_lateral = []
    delta_distance = 0.4
    #Agregar puntos a la lista
    for P in points3d_borde:
        #Definicion limite superior
        b_lim = abs((np.dot(P, normal) + D)) / delta_distance
        #Creacion de indices a lo largo de la linea recta
        indexes_line = np.arange(0, int(b_lim)  + 1, 1)
        #Crear puntos a lo largo de la linea y se agregan a la nube de puntos laterales
        points_in_line = P + (indexes_line[:, None]*delta_distance)*normal
        list_points_lateral.append(points_in_line)
    #Concatenacion de puntos para crear nube de puntos
    points3d_lateral = np.concatenate(list_points_lateral, axis=0)
    #####----------------------Union de nube de puntos----------------------#######
    points3d_total = np.vstack((points3d_segmented, points3d_plane, points3d_lateral))
    # Crear objeto PointCloud de Open3D
    points_cloud = o3d.geometry.PointCloud()
    points_cloud.points = o3d.utility.Vector3dVector(points3d_total)
   
    #Retornar
    return points_cloud



def points_cloud3d_to_mesh(points_cloud):
    """Funcion para convertir una nube de puntos a una malla 3d
    Input:
        points_cloud: Nube de puntos a convertir a una malla 3d
    Output:
        Malla 3d 
    """""
    mesh = 1
    return mesh
    