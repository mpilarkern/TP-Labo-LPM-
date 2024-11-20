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
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
import math
from matplotlib.colors import BoundaryNorm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from itertools import combinations



#%% Cargar datos
carpeta = "C:\\Users\\pilik\\OneDrive\\Escritorio\\Info académica Pili\\LCD - Pili\\2024\\2° cuatri\\laboratorio de datos\\TP Labo 2\\"
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
n_digit = 780
image_array = data_imgs[n_digit,:,:,0]
image_label = data_chrs[n_digit]


# Ploteo el grafico
plt.figure(figsize=(10,8))
plt.imshow(image_array, cmap='gray')
plt.title('caracter: ' + str(image_label))
plt.axis('off')  
plt.show()

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

#hacemos el heatmap de la matriz promedio de cada dígito
plt.figure(figsize=(8, 6))
sns.heatmap(promedio_0, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 0")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_1, annot=False, cmap='viridis')
plt.title("Matriz promedio del dígito 1")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_2, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 2")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_3, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 3")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_4, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 4")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_5, annot=False, cmap='viridis') 
plt.title("Matriz promedio del dígito 5")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_6, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 6")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_7, annot=False, cmap='viridis') 
plt.title("Matriz promedio del dígito 7")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_8, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 8")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(promedio_9, annot=False, cmap='viridis')  
plt.title("Matriz promedio del dígito 9")
plt.show()
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

def suma_fila(fila, matriz):
    res = 0
    for i in range(0,10):
        res = res + matriz[fila][i]
    return res

suma_0 = suma_fila(0,matriz_prom_dif)
suma_1 = suma_fila(1,matriz_prom_dif)
suma_2 = suma_fila(2,matriz_prom_dif)
suma_3 = suma_fila(3,matriz_prom_dif)
suma_4 = suma_fila(4,matriz_prom_dif)
suma_5 = suma_fila(5,matriz_prom_dif)
suma_6 = suma_fila(6,matriz_prom_dif)
suma_7 = suma_fila(7,matriz_prom_dif)
suma_8 = suma_fila(8,matriz_prom_dif)
suma_9 = suma_fila(9,matriz_prom_dif)

columnas = list(range(10))
valores = [suma_0, suma_1, suma_2, suma_3, suma_4, suma_5, suma_6, suma_7, suma_8, suma_9]

plt.bar(columnas, valores)

plt.xlabel('Dígitos')
plt.ylabel('Suma promedio diferenciabilidades')
plt.title('Comparación diferenciabilidades')
plt.xticks(columnas)

plt.show() 
#%% 
#¿Son todas las imágenes de una misma clase muy similares entre sí?

# Elegimos el dígito 3 para responder esta pregunta.
'''
def primeras_n_matrices(matrices, n):
    res = []
    for i in range(0,n):
        res.append(matrices[i])
    return res

#tomamos las primeras 100 imagenes de la clase del 3 como muestra representativa
recorte_3 = primeras_n_matrices(matrices_3,100)
#Primero tomamos todas las imágenes del 3 y creamos la matriz de pixeles iguales de esta clase:
matriz_pixeles_iguales_3 = matriz_pixeles_iguales(recorte_3)

plt.figure(figsize=(8, 6))
sns.heatmap(matriz_pixeles_iguales_3, annot=False, cmap='viridis')  
plt.title("matrices 3")
plt.show()'''
#%%
#Los árboles de decisión requieren un array 2D en el que cada fila sea una muestra (imagen) y cada columna sea una característica (píxeles aplanados o valores derivados).



#Elimino las dimensiones adicionales
data_imgs = data_imgs.squeeze()
data_chrs = data_chrs.squeeze()


#Me quedo con los dígitos que nos corresponden
digitos = [3, 4, 6, 7, 9]

filtro = np.isin(data_chrs, digitos) #filtro las etiquetas correspondientes a los digitos que quiero

data_imgs_filtradas = data_imgs[filtro]
data_chrs_filtradas = data_chrs[filtro]

#Aplano las imagenes
num_muestras, ancho, alto = data_imgs_filtradas.shape  # Obtener dimensiones
X = data_imgs_filtradas.reshape(num_muestras, ancho * alto) # (10000, 28*28) -> (10000, 784)

#Normalizo las imágenes
X = X / 255.0  # Escalar los valores de los píxeles entre 0 y 1

#Uso las etiquetas como Y
y = data_chrs_filtradas

#Divido en datos de entrenamiento y prueba
X_dev, X_heldout, y_dev, y_heldout = train_test_split(X, y, test_size=0.6, random_state=42)

print(f"Entrenamiento: {X_dev.shape}, {y_dev.shape}")
print(f"Prueba: {X_heldout.shape}, {y_heldout.shape}")

def distintasProfundidades(kf,criterio,profundidades,x_dev,y_dev):
    accuracy = []
    for i,(train_index, test_index) in enumerate(kf.split(x_dev)):
        kf_X_train, kf_X_test = x_dev[train_index], x_dev[test_index]
        kf_y_train, kf_y_test = y_dev[train_index], y_dev[test_index]  
        accuracy_list = []
        for pmax in profundidades:    
           arbol = tree.DecisionTreeClassifier(max_depth = pmax, criterion= criterio)
           arbol.fit(kf_X_train, kf_y_train)  
           pred = arbol.predict(kf_X_test)    
           acc = metrics.accuracy_score(kf_y_test, pred)
           accuracy_list.append(acc)
        accuracy.append(accuracy_list)    
    accuracy_np = np.array(accuracy)
    return np.mean(accuracy_np, axis=0)

profundidades = range(1,21)
nsplits = 5
kf = KFold(n_splits=nsplits)

accuracy_entropy = distintasProfundidades(kf,'entropy',profundidades,X_dev,y_dev)
accuracy_gini = distintasProfundidades(kf,'gini',profundidades,X_dev,y_dev)
#%%
plt.figure(figsize=(8, 6))
plt.scatter(range(1,len(accuracy_entropy)+1), accuracy_entropy, color='blue', label='entropy')
plt.scatter(range(1,len(accuracy_gini)+1), accuracy_gini, color='red', label='gini')

plt.xlabel('profundidad')
plt.ylabel('accuracy')

plt.ylim(0.2,1)
plt.xticks(range(21))
plt.xlim(0.5,21)
plt.title('Accuracy entre diferentes hiperparámetros')

plt.legend()  
plt.plot()
#%%
#Creo y entreno el árbol de decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_dev, y_dev)

# hago predicciones y evaluo el modelo
y_pred = modelo.predict(X_heldout)
precision = accuracy_score(y_heldout, y_pred)
print(f"Precisión del modelo: {precision}")

#visualizo la muestra con su predicción
idx = 475  # Índice de una imagen de prueba
plt.imshow(X_heldout[idx].reshape(ancho, alto), cmap='gray')
plt.title(f"Etiqueta real: {y_heldout[idx]}, Predicción: {y_pred[idx]}")
plt.show()

plt.figure(figsize=(20, 10))  # Ajustar tamaño de la figura
plot_tree(
    modelo,
    feature_names=[f"Pixel {i}" for i in range(X.shape[1])],  # Opcional: nombres de características
    class_names=[str(c) for c in modelo.classes_],  # Etiquetas de las clases
    filled=True,  # Colorear nodos según pureza
    rounded=True  # Bordes redondeados
)
plt.title("Árbol de Decisión")
plt.show()
#%%

def etiquetas_n(n):
    res = []
    for i in range(0, len(data_imgs)):
        if data_chrs[i] == n:
            res.append(i)
    return res

etiquetas_3 = etiquetas_n(3)


#%% Preprocesamiento de datos

from joblib import Parallel, delayed  # Para paralelización

def preprocesar_datos(data_imgs, data_chrs, digitos):
    """
    Filtra las imágenes y etiquetas, aplana las imágenes y las normaliza.
    """
    # Eliminar dimensiones adicionales
    data_imgs = data_imgs.squeeze()
    data_chrs = data_chrs.squeeze()

    # Filtrar los dígitos deseados
    filtro = np.isin(data_chrs, digitos)
    data_imgs_filtradas = data_imgs[filtro]
    data_chrs_filtradas = data_chrs[filtro]

    # Aplanar imágenes y normalizarlas
    num_muestras, ancho, alto = data_imgs_filtradas.shape
    X = data_imgs_filtradas.reshape(num_muestras, ancho * alto) / 255.0
    y = data_chrs_filtradas

    return X, y

#%% Validación cruzada paralelizada
def evaluar_profundidad(pmax, criterio, x_train, y_train, x_test, y_test):
    """
    Evalúa un árbol de decisión para una profundidad específica.
    """
    arbol = DecisionTreeClassifier(max_depth=pmax, criterion=criterio)
    arbol.fit(x_train, y_train)
    pred = arbol.predict(x_test)
    return metrics.accuracy_score(y_test, pred)

def distintasProfundidades(kf, criterio, profundidades, X_dev, y_dev):
    """
    Realiza validación cruzada para diferentes profundidades y calcula el accuracy promedio.
    """
    accuracy = []

    # Precomputar divisiones de KFold
    folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X_dev)]
    
    # Iterar sobre profundidades
    for pmax in profundidades:
        fold_accuracies = Parallel(n_jobs=-1)(  # Paralelizar el cálculo
            delayed(evaluar_profundidad)(
                pmax, criterio, 
                X_dev[train_idx], y_dev[train_idx], 
                X_dev[test_idx], y_dev[test_idx]
            )
            for train_idx, test_idx in folds
        )
        accuracy.append(np.mean(fold_accuracies))
    return accuracy

#%% Main
# Parámetros iniciales
digitos = [3, 4, 6, 7, 9]
profundidades = range(1, 21)
nsplits = 5

# Preprocesar datos
X, y = preprocesar_datos(data_imgs, data_chrs, digitos)

# Dividir datos en entrenamiento y prueba
X_dev, X_heldout, y_dev, y_heldout = train_test_split(X, y, test_size=0.6, random_state=42)

# Inicializar KFold
kf = KFold(n_splits=nsplits)

# Calcular accuracy para ambos criterios
accuracy_entropy = distintasProfundidades(kf, 'entropy', profundidades, X_dev, y_dev)
accuracy_gini = distintasProfundidades(kf, 'gini', profundidades, X_dev, y_dev)

#%% Visualización
plt.figure(figsize=(8, 6))
plt.scatter(profundidades, accuracy_entropy, color='blue', label='entropy')
plt.scatter(profundidades, accuracy_gini, color='red', label='gini')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.ylim(0.2, 1)
plt.xticks(range(1, 21))
plt.title('Accuracy entre diferentes hiperparámetros')
plt.legend()
plt.show()


#%%
'''
#Los árboles de decisión requieren un array 2D en el que cada fila sea una muestra (imagen) y cada columna sea una característica (píxeles aplanados o valores derivados).



#Elimino las dimensiones adicionales
data_imgs = data_imgs.squeeze()
data_chrs = data_chrs.squeeze()


#Me quedo con los dígitos que nos corresponden
digitos = [3, 4, 6, 7, 9]

filtro = np.isin(data_chrs, digitos) #filtro las etiquetas correspondientes a los digitos que quiero

data_imgs_filtradas = data_imgs[filtro]
data_chrs_filtradas = data_chrs[filtro]

#Aplano las imagenes
num_muestras, ancho, alto = data_imgs_filtradas.shape  # Obtener dimensiones
X = data_imgs_filtradas.reshape(num_muestras, ancho * alto) # (10000, 28*28) -> (10000, 784)

#Normalizo las imágenes
X = X / 255.0  # Escalar los valores de los píxeles entre 0 y 1

#Uso las etiquetas como Y
y = data_chrs_filtradas

#Divido en datos de entrenamiento y prueba
X_dev, X_heldout, y_dev, y_heldout = train_test_split(X, y, test_size=0.6, random_state=42)

print(f"Entrenamiento: {X_dev.shape}, {y_dev.shape}")
print(f"Prueba: {X_heldout.shape}, {y_heldout.shape}")

def distintasProfundidades(kf,criterio,profundidades,x_dev,y_dev):
    accuracy = []
    for i,(train_index, test_index) in enumerate(kf.split(x_dev)):
        kf_X_train, kf_X_test = x_dev[train_index], x_dev[test_index]
        kf_y_train, kf_y_test = y_dev[train_index], y_dev[test_index]  
        accuracy_list = []
        for pmax in profundidades:    
           arbol = tree.DecisionTreeClassifier(max_depth = pmax, criterion= criterio)
           arbol.fit(kf_X_train, kf_y_train)  
           pred = arbol.predict(kf_X_test)    
           acc = metrics.accuracy_score(kf_y_test, pred)
           accuracy_list.append(acc)
        accuracy.append(accuracy_list)    
    accuracy_np = np.array(accuracy)
    return np.mean(accuracy_np, axis=0)

profundidades = range(1,21)
nsplits = 5
kf = KFold(n_splits=nsplits)

accuracy_entropy = distintasProfundidades(kf,'entropy',profundidades,X_dev,y_dev)
accuracy_gini = distintasProfundidades(kf,'gini',profundidades,X_dev,y_dev)
'''
