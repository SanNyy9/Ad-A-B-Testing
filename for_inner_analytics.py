"""
Бизнес-вопрос:
    Влияет ли креатив на долю ответов "Yes" среди ответивших?

Метрика:
    p = yes / (yes + no) — доля "Yes" среди ответивших.

Гипотезы:
    H0: p_control = p_exposed  (креатив не меняет долю "Yes")
    H1: p_control != p_exposed (креатив меняет долю "Yes")

Уровень значимости (условный): alpha = 0.05
"""

import pandas as pd
from scipy import stats

main_dataset = pd.read_csv('ad_smart_data.csv')

'''Функция выводящая всю важную информацию датасета'''
def show_full_info(dataset):
    dataset.info()
    print()
    unique_number = dataset.nunique()
    print(unique_number)

'''Функция подсчета суммы для определенной категории'''
def sum_answers(dataset, column_grouped, agg_column):
    return dataset.groupby(column_grouped)[agg_column].sum()


sum_yes_grouped = sum_answers(main_dataset, 'experiment', 'yes') #Группировка "Yes" по control и exposed
sum_no_grouped = sum_answers(main_dataset, 'experiment', 'no') #Группировка "No" по control и exposed


answered_control = sum_yes_grouped['control'] + sum_no_grouped['control'] #Общее число ответивших в control
answered_exposed = sum_yes_grouped['exposed'] + sum_no_grouped['exposed'] #Общее число ответивших в exposed

only_yes_control = sum_yes_grouped['control'] #Ответившие "Yes" в control
only_yes_exposed = sum_yes_grouped['exposed'] #Ответившие "Yes" в exposed

proportion_yes_control = only_yes_control / answered_control #Доля ответивших "Yes" в control
proportion_yes_exposed = only_yes_exposed / answered_exposed #Доля ответивших "Yes" в exposed

delta = proportion_yes_exposed - proportion_yes_control

param_1 = proportion_yes_control*(1-proportion_yes_control) / answered_control #Оценка дисперсии оценки доли "Yes" для control
param_2 = proportion_yes_exposed * (1 - proportion_yes_exposed) / answered_exposed #Оценка дисперсии оценки доли "Yes" для exposed

se = (param_1 + param_2)**0.5 #Стандартная ошибка разницы долей для CI

margin_of_error = 1.96 * se

delta_negative = round(delta - margin_of_error, 3) #Левая граница доверительного интервала
delta_positive = round(delta + margin_of_error, 3) #Правая граница доверительного интервала

table = pd.DataFrame(
    {
        'Yes': [sum_yes_grouped['control'], sum_yes_grouped['exposed']],
        'No': [sum_no_grouped['control'],  sum_no_grouped['exposed']]
    },
    index=['control', 'exposed']
) #Датафрейм с общим количеством ответивших "Yes" и "No"

chi2, p_value, dof, expected = stats.chi2_contingency(table)

print('Доверительным интервалом в данной выборке будет являться промежуток:', f'[{delta_negative}:{delta_positive}]', sep='\n')
print('В диапозон доверительного интервала будет входить 0, изменения эффективности может и не быть')
print(f'p-value для данной выборки будет являться {p_value}')

"""
Вывод: 
1. 95% доверительный интервал для разницы долей Yes (exposed − control) составил [-0.037; 0.074].
Интервал содержит 0, поэтому у нас нет оснований отвергнуть гипотезу об отсутствии эффекта. 
Точечная оценка разницы близка к нулю, но ширина интервала допускает как небольшое снижение, 
так и небольшой рост доли Yes.

2. p-value = 0.556 > α = 0.05, поэтому мы не отвергаем H0.
Если бы эффекта не было, вероятность увидеть наблюдаемую разницу или более экстремальную составила бы 55.6%.
Это означает, что наблюдаемая разница хорошо объясняется случайным разбросом выборки.

3. По итогам теста статистически значимого влияния креатива 
на долю Yes среди ответивших не обнаружено (p = 0.556, 95% CI [-0.037; 0.074]).
При этом ширина интервала не позволяет исключить небольшой эффект — как положительный, так и отрицательный.
Для уверенного детектирования эффекта порядка нескольких п.п. потребуется больший объём данных.
"""
