import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


# Преобразуем "красивую" функцию в шумную.
def make_a_noise(fun):
    noise = np.random.default_rng(seed=6445)
    return fun + noise.uniform(-0.25, 0.25, len(fun))


# Выдает расстояние в «Манхеттенской» метрике.
def get_dist(a, b):
    return np.sum(np.abs(np.subtract(a, b)))


# Выдает уровень шума.
def get_noise(Y):
    return get_dist(Y[1:], Y[:-1])


# Выдает уровень отличия от исходного сигнала.
def get_distinction(a, b):
    return get_dist(a, b) / len(a)


# Создает картинку с точками.
def make_points(el_noise, el_dist, size):
    plt.xlabel('Noise')
    plt.ylabel('Distinction')
    plt.grid(linestyle='--')

    plt.xscale('log')

    colors = np.random.rand(len(el_noise))

    plt.scatter(el_noise, el_dist, c=colors)

    plt.savefig('PointsGraph_size_' + str(size) + '.png')

    plt.clf()


# Создает картинку с графиками функций.
def make_graph(X, Y, Y_with_noise, processed_Y, size):
    plt.title('Functions')
    plt.grid('-')

    plt.xlabel('X')
    plt.ylabel('f(X)')

    plt.plot(X, Y, color='blue')
    plt.plot(X, Y_with_noise, color='darkorange')
    plt.plot(X, processed_Y, color='green')

    plt.legend(['f(x) = sin(x) + 0.5', 'noise', 'filtering'])

    plt.savefig('MathGraph_size_' + str(size) + '.png')

    plt.clf()


def will_work(X, Y, Y_with_noise, size, N, lambda_l, ):
    random.seed(6445)

    # Создаем таблицу с данными.
    table = []

    # Заполняем её.
    for i in lambda_l:
        table.append([i, *main_search(Y_with_noise, size, N, i)])

    table_footer = ['h', 'alpha', 'w', 'd', 'J']
    table = pd.DataFrame(data=table, columns=table_footer)

    dist = np.round(np.abs(table['w'].values) + np.abs(table['d'].values), 4)
    table['dis'] = dist
    min_index = np.argmin(dist)

    # Построение графика с точками w ~ d.
    make_points(table['w'].values, table['d'].values, size)

    # Запись таблицы в файл.
    table.to_csv('Table_size_' + str(size) + '.csv')

    # Построение графика нормальной функции,
    # "шумной" функции и "выровненной".
    weights = table.loc[min_index]['alpha']
    processed_Y = arithmetically_glide(Y_with_noise, weights)
    make_graph(X, Y, Y_with_noise, processed_Y, size)


# Ищем вектор весов для данной длины.
def search_for_weights(size):
    # Массив в центральной точкой.
    res = [random.uniform(0, 1)]

    # Все точки, кроме последних.
    for _ in range(0, (size - 1) // 2 - 1):
        tmp = 0.5 * random.uniform(0, 1 - sum(res))
        res = [tmp, *res, tmp]

    # Добавляем последние точки.
    tmp = 0.5 * (1 - sum(res))
    res = [tmp, *res, tmp]

    return res


# Создаем "строку" готовых данных.
def main_search(Y_with_noise, size, N, lambda_i):
    # Создаем массив весов.
    weights = [search_for_weights(size) for i in range(N)]

    # Массив обработанных значений функции.
    processed_Y = [arithmetically_glide(Y_with_noise, i) for i in weights]

    # Уровень «зашумленности».
    level_of_noise = [get_noise(i) for i in processed_Y]

    # Уровень отличия от исходного сигнала.
    level_of_distinction = [get_distinction(i, Y_with_noise) for i in processed_Y]

    # Функционал.
    functional = [lambda_i * i + (1. - lambda_i) * j
                  for i, j in zip(level_of_noise, level_of_distinction)]

    # Ищем индекс минального функционала.
    index = np.argmin(functional)

    res = [np.round(weights[index], 4), np.round(level_of_noise[index], 4),
           np.round(level_of_distinction[index], 4), np.round(functional[index], 4)]
    return res


# Функция проезда по массиву шумной функции.
def arithmetically_glide(Y_with_noise, Weights):
    # Длина крыса каретки.
    wingLen = (len(Weights) - 1) // 2

    # Добавляем "нулёвые" крылья в массив Y_with_noise.
    Y_with_wings = np.pad(Y_with_noise, (wingLen, wingLen),
                          'constant', constant_values=0)

    # Проежаемся кареткой по массиву.
    return [np.sum(Y_with_wings[i:i + len(Weights)] *Weights)
            for i in range(len(Y_with_noise))]