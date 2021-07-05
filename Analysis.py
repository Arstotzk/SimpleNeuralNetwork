import numpy as np
import func
import var
from PIL import Image

weights = np.load("weights.npy")# Загружаем веса
new_image = Image.open("images/0.jpg") #Выбор изображения для анализа
norm_image = func.normalization(new_image, var.width, var.height) #Нормализация
inputValue = np.zeros(shape=(var.width,var.height), dtype=float)#Массив с входящими значениями
for i in range(0, var.width):#Проходим по изображению и записываем входящие значения
    for j in range(0, var.height):
        color = norm_image.getpixel((i, j))[0]
        inputValue[i][j] = -1 + (color/128)

massSigmods = np.zeros(shape=(len(var.neirons), var.neirons[0]), dtype=float)#Массив сигмоид
for n in range(0, len(var.neirons)):  # Номер слоя
    for i in range(0, var.neirons[0]):  # номер нейрона
        if i < var.neirons[n]:#Первый слой
            massSigmods[n][i] = func.sigmoid(inputValue, weights[0][i],norm_image)# Расчет сигмоиды
        if (n > 0) & (i < var.neirons[n]):#Следующие слои
            massSigmods[n][i] = func.sigmoid(massSigmods[n - 1], weights[n][i],norm_image)#Расчет сигмоиды
error2 = np.zeros(shape=(4,))#Сигмоиды на выходе
for i in range(0, 4):
    error2[i] = massSigmods[len(var.neirons) - 1][i]
print("Результат: ", error2)#Выводим сигмоиды на выходе