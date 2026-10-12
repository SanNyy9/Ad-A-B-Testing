import pandas as pd
from scipy.stats import norm


'''Функция выводящая всю важную информацию датасета'''
def show_full_info(dataset):
    dataset.info()
    print()
    unique_number = dataset.nunique()
    print(unique_number)

'''Функция подсчета суммы для определенной категории'''
def sum_answers(dataset, column_grouped, agg_column):
    return dataset.groupby(column_grouped)[agg_column].sum()


alpha = 0.05
power = 0.8

z_alpha = norm.ppf(1 - alpha / 2)
z_beta = norm.ppf(power)