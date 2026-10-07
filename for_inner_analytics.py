import pandas as pd

main_dataset = pd.read_csv('ad_smart_data.csv')

def sum_answers(dataset, column_grouped, agg_column):
    return dataset.groupby(column_grouped)[agg_column].sum()


sum_yes_grouped = sum_answers(main_dataset, 'experiment', 'yes')
sum_no_grouped = sum_answers(main_dataset, 'experiment', 'no')
sum_everyone_grouped = main_dataset.groupby('experiment')['auction_id'].count()
sum_everyone_answered_control = sum_yes_grouped['control'] + sum_no_grouped['control']
sum_everyone_answered_exposed = sum_yes_grouped['exposed'] + sum_no_grouped['exposed']

sum_unique_values = (main_dataset['auction_id']
                     .unique()
                     .size)

only_yes_control = sum_yes_grouped['control'].sum()
only_no_control = sum_no_grouped['control'].sum()

only_yes_exposed = sum_yes_grouped['exposed'].sum()
only_no_exposed = sum_no_grouped['exposed'].sum()

percent_yes_control = round(only_yes_control / sum_everyone_answered_control * 100, 2)
percent_yes_exposed = round(only_yes_exposed / sum_everyone_answered_exposed * 100, 2)

print(f'Изначально проголосовало "да": {percent_yes_control}%')
print(f'Затем проголосовало "да" {percent_yes_exposed}%')