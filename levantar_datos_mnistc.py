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
import seaborn as sns
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import math


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
#plt.show()

#%%

# Separamos las imágenes según el dígito al que pertenecen
def obtener_matrices_etiquetadas_n(n):
    res = []
    for i in range(0, 10000):
        image_array = data_imgs[i,:,:,0]
        image_label = data_chrs[i]
        if image_label == n:
            res.append(image_array)
    return res


# Para cada dígito creamos una matriz promedio (representa la imagen promedio de ese dígito)
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

# Vemos qué atributos son iguales en más imagenes

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

matriz = matriz_pixeles_iguales(lista_matrices)

#%%
#¿Cuáles parecen ser los atributos (i.e., píxeles) más relevantes para predecir el dígito al que corresponde la imagen? 

#  Heatmap de coincidencias en pixeles entre cada par de digitos
matriz = matriz_pixeles_iguales(lista_matrices)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=False, cmap='viridis')  # Puedes cambiar 'viridis' por otra escala de colores
plt.title("Coincidencias en cada pixel entre todos los pares de digitos")
plt.show()

#Los pixeles con más coincidencias son los atributos menos útiles pues no diferencian cada dígito
#Notamos que los pixeles de los bordes son los de menor diferenciabilidad.
#La diferenciabilidad aumenta a medida que nos acercamos al centro de la imagen, por lo que los atributos del centro tienen mayor informcación

#Hacemos un histograma separando los atributos en grupos según su distancia al centro para verificar nuestras observaciones primarias

def distancia_a_centro(a,b,centro):
    x = centro[0] - a
    y = centro[1] - b
    return math.sqrt(x*x+y*y)

def promedio_lista(lista):
    suma = 0
    for i in range (0, len(lista)):
        suma = suma + lista[i]
    return suma/len(lista)

centro = (12, 13)

def pixeles_segun_distancia_al_centro(matriz,centro):
    borde = []
    dis_1 = []
    dis_2 = []
    dis_3 = []
    for i in range (0,28):
        for j in range (0,28):
            if (distancia_a_centro(i,j,centro) >= 13):
                borde.append(matriz[i,j])
            elif (distancia_a_centro(i,j,centro) >= 9):
                dis_1.append(matriz[i,j])
            elif (distancia_a_centro(i,j,centro) >= 5):
                dis_2.append(matriz[i,j])
            else:
                dis_3.append(matriz[i,j])
    return [promedio_lista(borde),promedio_lista(dis_1),promedio_lista(dis_2),promedio_lista(dis_3)]

df = pd.DataFrame([pixeles_segun_distancia_al_centro(matriz, centro)], columns= ["d >= 13","13 > d >= 9", "9 > d >= 5", "5 > d" ])


# Hacemos el gráfico de barras

valores = df.iloc[0] 
columnas = df.columns

plt.bar(columnas, valores)

plt.xlabel('Pixeles a distancia "d" del centro')
plt.ylabel('Promedio de coincidencias por pixel')
plt.title('Promedio de coincidencias de color de pixel según ubicación en la imagen respecto al centro')

plt.show()                

# Como muestra este gráfico, los valores del borde de la imagen tienen mucha más coincidencia entre las imagenes de los distintos digitos
# A medida que nos acercamos al centro de la imagen los pixeles coinciden menos entre imágenes.
# Vemos que los pixeles a distancia 9 o menor del centro de la imagen son los que más permiten diferenciar entre dígitos
#%%

#NO SE USA

def matriz_pixeles_12(matriz):
    matriz_resultado = np.zeros((28, 28))
    for i in range (0,28):
            for j in range (0,28):
                valor_pixel = matriz[i,j]
                if (valor_pixel <=  14):
                    matriz_resultado[i,j] = 1
    return matriz_resultado

matriz_14 = matriz_pixeles_12(resultado)

matriz_1479 = matriz_pixeles_iguales([promedio_1,promedio_4,promedio_7,promedio_9])

matriz_0268 = matriz_pixeles_iguales([promedio_0,promedio_2,promedio_6,promedio_8])
    
#%%
#Hay digitos parecidos entre si?

matriz_comparacion_0_1 = matriz_pixeles_iguales([promedio_1,promedio_0])

matriz_comparacion_1_7 = matriz_pixeles_iguales([promedio_1,promedio_7])

def matriz_promedio_pixeles_iguales(matrices):
    matriz_resultado = np.zeros((10, 10))
    for a in range (0,10):
        for b in range (0,10):
            matriz_pixeles = matriz_pixeles_iguales([matrices[a],matrices[b]])
            promedio = np.mean(matriz_pixeles)
            matriz_resultado[a,b] = promedio * 25
    return matriz_resultado

# Dice el porcentaje de pixeles iguales entre las imágenes promedio de cada par de dígitos
matriz_porcentaje_coincidencias = matriz_promedio_pixeles_iguales(lista_matrices)

# Definir los límites personalizados para evitar que la diagonal dificulte la visualización
limites = [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 100]  # Último rango 79 y 100 tienen mismo color
colores = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
           '#2171b5', '#08519c', '#08306b', '#08306b', '#08306b', '#08306b']  # Escala azul

# Creamos el heatmap
cmap = ListedColormap(colores)
norm = BoundaryNorm(limites, cmap.N)


plt.figure(figsize=(10, 8))
sns.heatmap(matriz_porcentaje_coincidencias, annot=True, fmt='0.1f', cmap=cmap, norm=norm, cbar=True)  # fmt='d' para enteros


plt.title("Porcentaje de coincidencias de pixeles")
plt.show()

#Esta visualización muestra el porcentaje de coincidencias de los pixeles para cada par de dígitos

# Por ejemplo, el 0 y el 1 coinciden solo en el 70% de sus pixeles por lo que son relativamente faciles de diferenciar
# El 5 y el 6 coinciden en el 72% de sus pixeles por lo que es un poco más dificil diferenciarlos


#%%
# Además del porcentaje de coincidencia, es importante notar que para una comparación en particular hay pixeles específicos que diferencian un dígito de otro

# IDEAS: calcular para cada par de dígitos la cantidad de pixeles específicos considerando una diferencia minima entre el pixel de cada imagen para considerarlo especifico
# IDEAS: calcular para cada par de dígitos la especificidad de todos sus pixeles (la diferencia entre los valores de cada pixel en cada imagen)
# IDEAS: calcular para cada par de dígitos la especificidad de sus pixeles específicos (cuan buenos son esos pixeles para determinar el dígito)

def cantidad_de_pixeles_especificos(matriz1, matriz2, umbral):
    cantidad = 0
    for i in range (0,28):
            for j in range (0,28):
                diferencia_valor_pixel = abs(matriz1[i,j] - matriz2[i,j])
                if (diferencia_valor_pixel >  umbral):
                    cantidad = cantidad + 1
    return cantidad

def matriz_cantidad_pixeles_especificos(matrices, umbral_especificidad):
    matriz_resultado = np.zeros((10, 10))
    for a in range (0,10):
        for b in range (0,10):
            matriz_resultado[a,b] = cantidad_de_pixeles_especificos(matrices[a], matrices[b], umbral_especificidad)
    return matriz_resultado

pixeles_especificos_0_1 = cantidad_de_pixeles_especificos(promedio_0, promedio_1, 120)

matriz_pixeles_especificos = matriz_cantidad_pixeles_especificos(lista_matrices, 100)

## podriamos elegir el umbral de especificidad volviendo a los datos sin promediar y viendo a cuantas imagenes diferencia con cada umbral

#  Heatmap de pixeles específicos entre cada par de digitos

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_pixeles_especificos, annot=True, cmap='cividis')  # Puedes cambiar 'viridis' por otra escala de colores
plt.title("Cantidad de pixeles deterministicos para cada par de dígitos")
plt.show()

# Como se ve en el heatmap, el 0 y 1 tienen 86 pixeles determinísticos mientras que el 5 y el 6 tienen solo 11 pixeles determinísticos
# Por lo que es mucho más sencillo diferenciar al 1 y 0 que al 5 y 6



def matriz_diferencia_de_pixeles(matriz1, matriz2):
    matriz_resultado = np.zeros((28, 28))
    for i in range (0,28):
        for j in range (0,28):
            valor1 = matriz1[i,j]
            valor2 = matriz2[i,j]
            matriz_resultado[i,j] = abs(valor1 - valor2)
    return matriz_resultado

matriz_dif = matriz_diferencia_de_pixeles(promedio_0, promedio_1)

def matriz_promedio_diferenciabilidad_de_pixeles(matrices):
    matriz_resultado = np.zeros((10, 10))
    for a in range (0,10):
        for b in range (0,10):
            matriz_resultado[a,b] = np.mean(matriz_diferencia_de_pixeles(matrices[a], matrices[b]))
    return matriz_resultado

matriz_prom_dif = matriz_promedio_diferenciabilidad_de_pixeles(lista_matrices)

# Definir los límites personalizados para evitar que la diagonal dificulte la visualización
limites = [0, 8, 10, 13, 16,18, 20, 22, 24, 26, 28, 31]  # Último rango 79 y 100 tienen mismo color
colores = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6',
           '#2171b5', '#08519c', '#08306b', '#08306b', '#08306b']  # Escala azul

# Creamos el heatmap
cmap = ListedColormap(colores)
norm = BoundaryNorm(limites, cmap.N)


plt.figure(figsize=(10, 8))
sns.heatmap(matriz_prom_dif, annot=True, fmt='0.1f', cmap=cmap, norm=norm, cbar=True) 


plt.title("Promedio de diferenciabilidad de imagenes")
plt.show()

#%% 
#¿Son todas las imágenes de una misma clase muy similares entre sí?

# Elegimos el dígito 3 para responder esta pregunta.

#Primero tomamos todas las imágenes del 3 y creamos la matriz de pixeles iguales de esta clase:
matriz_pixeles_iguales_3 = matriz_pixeles_iguales(matrices_3) #no anda
