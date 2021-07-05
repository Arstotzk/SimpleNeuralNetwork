import numpy as np
import random
import math
import func
import var
from PIL import Image

for i in range(0, var.numImage):#Нормализация всех изображений
    new_image = Image.open("images/" + str(i) + ".jpg")#Перебор изображений
    var.massImageNorm[i] = func.normalization(new_image, var.width, var.height)#Нормализация изображения и запись его в массив
weights = np.zeros(var.neirons[0]*var.width*var.height*len(var.neirons)).reshape(len(var.neirons),var.neirons[0],var.width*var.height) #Массив весов
ni = 1 #Модуль задающий скорость "движения"
for l in range(0, len(var.neirons)):#Слой
    for n in range(0, var.neirons[0]):#Нейрон
        for i in range(0, var.width*var.height):#Номер веса
            weights[l][n][i] = random.uniform(-0.5, 0.5) #Генерация весов
inputValue = np.zeros(shape=(var.width,var.height), dtype=float)#Массив входящих значений с изображения
orderImage = 0 # Номер изображения для пересчета весов
errorDef2 = 0 # Переменная для средней ошибки по всем изображениям
for g in range (0, 2000):# Начало обучения
    for i in range(0, var.width):#Проходим по изображению и записываем входяшие значения
        for j in range(0, var.height):
            color = var.massImageNorm[orderImage].getpixel((i, j))[0]
            inputValue[i][j] = -1 + (color/128)

    massSigmods = np.zeros(shape=(len(var.neirons), var.neirons[0]), dtype=float)#Массив сигмоид
    for n in range (0, len(var.neirons)):#Номер слоя
        for i in range (0, var.neirons[0]):#номер нейрона
            if i < var.neirons[n]:#Для первого слоя
                massSigmods[n][i] = func.sigmoid(inputValue,weights[0][i],var.massImageNorm[0])#Расчет сигмоиды
            if (n > 0) & (i < var.neirons[n]):#Для следующих слове
                massSigmods[n][i] = func.sigmoid(massSigmods[n-1], weights[n][i], var.massImageNorm[0])#Расчет сигмоиды
    exit1 = [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]#Выходные значения
    error2 = np.zeros(shape=(var.neirons[2],))#Ошибка на последнем слое
    error1 = np.zeros(shape=(var.neirons[1],))#Ошибка на втором слое
    error0 = np.zeros(shape=(var.neirons[0],))#Ошибка на первом слое
    error = []#Массив для ошибок
    errorDef = 0 # Среднее значение ошибки для изображения
    for i in range(0, var.neirons[2]):#Проходим по последнему слою нейронов
        error2[i] = massSigmods[len(var.neirons)-1][i] - exit1[orderImage][i] #Расчет ошибки на последнем слое
        errorDef += error2[i] ** 2
    errorDef = math.sqrt(errorDef)#Расчитываем среднее значение ошибки для изображения
    for i in range(0, var.neirons[1]):#Расчет ошибки для 2 слоя
        error1[i] = func.net(error2,weights[1][i],var.massImageNorm[0])
    for i in range(0, var.neirons[0]):#Расчет ошибки для 1 слоя
        error0[i] = func.net(error1,weights[0][i],var.massImageNorm[0])
    error.append(error0)#Добавляем в масиив ошибки все ошибки по слоям
    error.append(error1)
    error.append(error2)
    print("Итерация:", g, "Ошибка: ", error2)

    for l in range (0, len(var.neirons)):#Слой
        for n in range (0, var.neirons[l]):#нейрон
            num = 0
            if l == 0:#Первый слой
                for i in range (0, var.width):#Проходим по ширине
                    for j in range(0, var.height):#Проходим по высоте
                        newValue = (error[l][n] * massSigmods[l][n] * (1-massSigmods[l][n]) * inputValue[i][j] * ni)#Считаем коррекцию веса
                        weights[l][n][num] -= newValue#Корректируем вес
                        num += 1
            else:#Остальные слои
                for num in range(0, var.neirons[l-1]):#Проходим по нейронам из предидущего слоя
                    newValue = (error[l][n] * massSigmods[l][n] * (1 - massSigmods[l][n]) * ni * massSigmods[l-1][num])#Считаем коррекцию веса
                    weights[l][n][num] -= newValue#Корректируем вес
    errorDef2 += errorDef #Добавляем в переменную среднего значения ошибки по всем изображениям значение средней ошибки по текушему изображению
    orderImage += 1 #Переключаемся на следующее изображение
    if orderImage == 4:#Если переключились на 5 изображение
        orderImage = 0#Переключаемся на первое изображение
        print("Средняя ошибка:", errorDef2/4)#Выводим среднее значение ошибки по всем изображениям
        if ((errorDef2/4) < 0.1):#Если ошибка получилась меньше 0.1, то заканчиваем обучение
            break
        errorDef2 = 0#Обнуляем значение средней ошибки
np.save("weights",weights)#Сохраняем получившиеся веса