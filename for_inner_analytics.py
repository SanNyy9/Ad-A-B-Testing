import pandas as pd

main_dataset = pd.read_csv('ad_smart_data.csv')

def sum_answers(dataset, column_grouped, agg_column):
    return dataset.groupby(column_grouped)[agg_column].sum()

# def avg_mistake(dataframe, grouped_name, param, A, B):
#     avg_nums = dataframe.groupby(grouped_name)[param].mean()
#     delta = (dataframe[param] - avg_nums[ab])**2
#     sd = (delta.sum() / delta.size)**0.5
#     se = sd / (delta.size) ** 0.5
#     return se



sum_yes_grouped = sum_answers(main_dataset, 'experiment', 'yes')
sum_no_grouped = sum_answers(main_dataset, 'experiment', 'no')
sum_everyone_grouped = main_dataset.groupby('experiment')['auction_id'].count()
n1 = sum_yes_grouped['control'] + sum_no_grouped['control']
n2 = sum_yes_grouped['exposed'] + sum_no_grouped['exposed']

sum_unique_values_n2 = (main_dataset['auction_id']
                     .unique()
                     .size)

only_yes_control = sum_yes_grouped['control'].sum()
only_no_control = sum_no_grouped['control'].sum()

only_yes_exposed = sum_yes_grouped['exposed'].sum()
only_no_exposed = sum_no_grouped['exposed'].sum()

p1 = only_yes_control / n1
p2 = only_yes_exposed / n2

delta = p2 - p1


param_1 = p1*(1-p1) / n1
param_2 = p2 * (1 - p2) / n2

se = (param_1 + param_2)**0.5

print(delta + 1.96 * se)
print(delta - 1.96 * se)

