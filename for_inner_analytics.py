import pandas as pd


main_dataset = pd.read_csv('ad_smart_data.csv')

def show_full_info(dataset):
    dataset.info()
    print()
    unique_number = dataset.nunique()
    print(unique_number)

def sum_answers(dataset, column_grouped, agg_column):
    return dataset.groupby(column_grouped)[agg_column].sum()

sum_yes_grouped = sum_answers(main_dataset, 'experiment', 'yes')
sum_no_grouped = sum_answers(main_dataset, 'experiment', 'no')

n1 = sum_yes_grouped['control'] + sum_no_grouped['control']
n2 = sum_yes_grouped['exposed'] + sum_no_grouped['exposed']

only_yes_control = sum_yes_grouped['control']
only_yes_exposed = sum_yes_grouped['exposed']

p1 = only_yes_control / n1
p2 = only_yes_exposed / n2

delta = p2 - p1

param_1 = p1*(1-p1) / n1
param_2 = p2 * (1 - p2) / n2

se = (param_1 + param_2)**0.5

delta_negative = round(delta - 1.96 * se, 3)
delta_positive = round(delta + 1.96 * se, 3)


print('Доверительным интервалом в данной выборке будет являться промежуток:', f'[{delta_negative}:{delta_positive}]', sep='\n')
print('В диапозон доверительного интервала будет входить 0')

show_full_info(main_dataset)