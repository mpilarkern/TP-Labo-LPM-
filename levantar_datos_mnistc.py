# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:24:31 2024

@author: milen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:39:20 2024

@author: rodrigo
"""

# script para cargar y plotear dígitos


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#%% Cargar datos
carpeta = "C:\\Users\\milen\\OneDrive\\Documents\\Trabajos 2024\\Facultad\\Laboratorio de Datos\\tp2\\"
# un array para las imágenes, otro para las etiquetas (por qué no lo ponen en el mismo array #$%@*)
data_imgs = np.load(carpeta + 'mnistc_images.npy')
data_chrs = np.load(carpeta + 'mnistc_labels.npy')[:,np.newaxis]


# mostrar forma del array:
# 1ra dimensión: cada una de las imágenes en el dataset
# 2da y 3ra dimensión: 28x28 píxeles de cada imagen
print(data_imgs.shape)
print(data_chrs.shape)



#%% Grafico imagen

# Elijo la imagen correspondiente a la letra que quiero graficar
n_digit = 5
image_array = data_imgs[n_digit,:,:,0]
image_label = data_chrs[n_digit]


# Ploteo el grafico
plt.figure(figsize=(10,8))
plt.imshow(image_array, cmap='gray')
plt.title('caracter: ' + str(image_label))
plt.axis('off')  
plt.show()

#%%
def obtener_matrices_etiquetadas_n(n):
    res = []
    for i in range(0, 10000):
        image_array = data_imgs[i,:,:,0]
        image_label = data_chrs[i]
        if image_label == n:
            res.append(image_array)
    return res

def matriz_de_promedios(lista_matrices):
    suma_matrices= np.sum(lista_matrices, axis=0)
    longitud = len(lista_matrices)
    res = suma_matrices/longitud
    return res

matrices_0= obtener_matrices_etiquetadas_n(0)
promedio_0 = matriz_de_promedios(matrices_0)

matrices_1 = obtener_matrices_etiquetadas_n(1)
promedio_1 = matriz_de_promedios(matrices_1)

matrices_2 = obtener_matrices_etiquetadas_n(2)
promedio_2 = matriz_de_promedios(matrices_2)

matrices_3 = obtener_matrices_etiquetadas_n(3)
promedio_3 = matriz_de_promedios(matrices_3)

matrices_4 = obtener_matrices_etiquetadas_n(4)
promedio_4 = matriz_de_promedios(matrices_4)

matrices_5 = obtener_matrices_etiquetadas_n(5)
promedio_5 = matriz_de_promedios(matrices_5)

matrices_6 = obtener_matrices_etiquetadas_n(6)
promedio_6 = matriz_de_promedios(matrices_6)

matrices_7 = obtener_matrices_etiquetadas_n(7)
promedio_7 = matriz_de_promedios(matrices_7)

matrices_8 = obtener_matrices_etiquetadas_n(8)
promedio_8 = matriz_de_promedios(matrices_8)

matrices_9 = obtener_matrices_etiquetadas_n(9)
promedio_9 = matriz_de_promedios(matrices_9)

#%%

umbral = 5

def matriz_pixeles_iguales(matrices):
    matriz_resultado = np.zeros((28, 28))
    for matriz1 in matrices:
        for matriz2 in matrices:
            for i in range (0,28):
                for j in range (0,28):
                    valor_res = matriz_resultado[i,j]
                    valor1 = matriz1[i,j]
                    valor2 = matriz2[i,j]
                    matriz_resultado[i,j] = valor_res + comparar_pixel(valor1,valor2)
    return matriz_resultado

def comparar_pixel(valor1, valor2):
    if (abs(valor1 - valor2) < umbral):
        return 1 
    else:
        return 0

lista_matrices = [promedio_0,promedio_1, promedio_2, promedio_3, promedio_4, promedio_5, promedio_6, promedio_7, promedio_8, promedio_9]
resultado = matriz_pixeles_iguales(lista_matrices)

matrices_9 = obtener_matrices_etiquetadas_n(9)
promedio_9 = matriz_de_promedios(matrices_9)

