"""
Лабораторная работа: Численные вычисления и анализ данных с использованием NumPy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict

# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """
    Создать массив от 0 до 9.

    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """
    Создать матрицу 5x5 со случайными числами [0,1].

    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Преобразовать вектор формы (10,) в матрицу (2,5).

    Args:
        vec (numpy.ndarray): Входной массив формы (10,).

    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Транспонирование матрицы.

    Args:
        mat (numpy.ndarray): Входная матрица.

    Returns:
        numpy.ndarray: Транспонированная матрица.
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное сложение двух векторов одинаковой длины.

    Args:
        a (numpy.ndarray): Первый вектор.
        b (numpy.ndarray): Второй вектор.

    Returns:
        numpy.ndarray: Результат поэлементного сложения.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: Union[float, int]) -> np.ndarray:
    """
    Умножение вектора на скаляр.

    Args:
        vec (numpy.ndarray): Входной вектор.
        scalar (float|int): Число для умножения.

    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр.
    """
    return scalar * vec


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Поэлементное умножение двух массивов одинаковой формы.

    Args:
        a (numpy.ndarray): Первый массив.
        b (numpy.ndarray): Второй массив.

    Returns:
        numpy.ndarray: Результат поэлементного умножения.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Вычисление скалярного произведения двух векторов.

    Args:
        a (numpy.ndarray): Первый вектор.
        b (numpy.ndarray): Второй вектор.

    Returns:
        float: Скалярное произведение.
    """
    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Умножение двух матриц.

    Args:
        a (numpy.ndarray): Первая матрица.
        b (numpy.ndarray): Вторая матрица.

    Returns:
        numpy.ndarray: Результат умножения матриц.
    """
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """
    Вычисление определителя квадратной матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица.

    Returns:
        float: Определитель матрицы.
    """
    return np.linalg.det(a)


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """
    Вычисление обратной матрицы.

    Args:
        a (numpy.ndarray): Квадратная матрица.

    Returns:
        numpy.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решение системы линейных уравнений Ax = b.

    Args:
        a (numpy.ndarray): Матрица коэффициентов A.
        b (numpy.ndarray): Вектор свободных членов b.

    Returns:
        numpy.ndarray: Решение системы x.
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """
    Загрузка данных из CSV-файла в массив NumPy.

    Args:
        path (str): Путь к CSV-файлу.

    Returns:
        numpy.ndarray: Загруженные данные.
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> Dict[str, float]:
    """
    Вычисление основных статистических показателей для одномерного массива.

    Args:
        data (numpy.ndarray): Одномерный массив данных.

    Returns:
        dict: Словарь со статистическими показателями:
            - mean: среднее арифметическое
            - median: медиана
            - std: стандартное отклонение
            - min: минимум
            - max: максимум
            - 25-perc: 25-й перцентиль
            - 75-perc: 75-й перцентиль
    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "25-perc": np.percentile(data, 25),
        "75-perc": np.percentile(data, 75)
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-Max нормализация массива в диапазон [0, 1].

    Формула: (x - min) / (max - min).

    Args:
        data (numpy.ndarray): Входной массив.

    Returns:
        numpy.ndarray: Нормализованный массив.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray) -> None:
    """
    Построение гистограммы распределения оценок.

    Args:
        data (numpy.ndarray): Одномерный массив оценок.
    """
    plt.clf()
    plt.hist(data, bins='auto', edgecolor='black')
    plt.xlabel("Оценка")
    plt.ylabel("Количество")
    plt.title("Гистограмма оценок")
    plt.savefig("plots/hist.png")
    plt.clf()


def plot_heatmap(matrix: np.ndarray) -> None:
    """
    Построение тепловой карты корреляционной матрицы.

    Args:
        matrix (numpy.ndarray): Квадратная матрица корреляций.
    """
    plt.clf()
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Корреляционная матрица")
    plt.savefig("plots/hmap.png")
    plt.clf()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    """
    Построение линейного графика зависимости оценок от номеров студентов.

    Args:
        x (numpy.ndarray): Номера студентов.
        y (numpy.ndarray): Оценки студентов.
    """
    plt.clf()
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title("Зависимость студент -> оценка")
    plt.xlabel("Студент")
    plt.ylabel("Оценка")
    plt.savefig("plots/line.png")
    plt.clf()


if __name__ == "__main__":
    print("Запустите python -m pytest test.py -v для проверки лабораторной работы.")