import math
from PIL import Image

def net(inputValue, weights, img):#Расчет значения нейрона\ошибки
    neir = 0 #Значение нейрона
    weightsInt = 0 #Переменная для перебора весов
    width, height = img.size #Получаем размеры по картинке
    if len(inputValue)==width:#Расчет нейронов для первого слоя
        for i in range (0, len(inputValue)):#Ширина
            for j in range (0, len(inputValue[0])):#Высота
                neir += inputValue[i][j] * weights[weightsInt] #Добаляем нейрону входяшее значение * вес
                weightsInt += 1
    else:#Расчет нейронов для следуюших слоев
        for i in range (0, len(inputValue)):
            neir += inputValue[i] * weights[i]
    return neir#Возвращаем значение нейрона
def sigmoid(inputValue, weights, img):#Расчет сигмоиды
    neir = net(inputValue,weights, img)#Расчитываем значение нейрона
    return float(1/ (1 + (math.e ** (-neir))))#Возвращаем значение сигмоиды
def normalization(imag, width, height):#Нормализация изображений
    img_new = imag.convert('RGB')#Конвернитуем в RGB
    img_new2 = imag.convert('RGB')
    img_new = img_new.resize((width, height), Image.ANTIALIAS)#Меняем размер изображений
    img_new2 = img_new2.resize((width, height), Image.ANTIALIAS)
    y = 0
    for i in range(0, width):#Преобразование в градации серого
        for j in range(0, height):
            r, g, b = img_new2.getpixel((i, j))
            y = int(0.299 * r + 0.587 * g + 0.144 * b)
            img_new.putpixel((i, j), (y, y, y))
    img_new = img_new.convert('YCbCr')#Коневернируем в YCbCr
    return img_new