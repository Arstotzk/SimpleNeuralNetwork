import numpy as np

numImage = 4 #Кол-во изображений
massImageNorm = np.empty(shape=(4,), dtype=object)

width, height = 30, 30#Размеры изображения после нормализации
neirons = [150,25,4] #Массив с кол-вом нейронов