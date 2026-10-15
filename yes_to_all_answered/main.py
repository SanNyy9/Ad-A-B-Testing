"""A/B-тест: влияет ли новый креатив
на долю ответов 'Yes' среди ответивших."""

from stats_utils import *

SEP = '--' * 40


def main():
    df = pd.read_csv('ad_smart_data.csv')

    shares, p_srm = check_srm(df)
    print('SRM-проверка')
    print(f'Соотношение control / exposed: {shares["control"]}% / {shares["exposed"]}%')
    print(f'p-value = {p_srm:.4f}')

    if p_srm < SRM_ALPHA:
        print('SRM обнаружен — результатам доверять нельзя.')
        return
    print('SRM не обнаружен — рандомизация корректна.')
    print(SEP)

    r = analyze_metric(df)
    print(f'Ответивших: control = {r["n_control"]}, exposed = {r["n_exposed"]}')
    print(f'Доля Yes: control = {r["p_control"]:.4f}, exposed = {r["p_exposed"]:.4f}')
    print(f'Точечная оценка delta = {r["delta"]:.5f}')
    print(SEP)
    print(f'MDE = {r["mde"] * 100:.4f} п.п.')
    print(f'95% CI для delta: [{r["ci_low"]}; {r["ci_high"]}]')
    print(f'p-value = {round(r["p_value"], 3)}')


if __name__ == '__main__':
    main()