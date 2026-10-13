"""Утилиты для A/B-анализа: SRM, MDE, доверительный интервал, хи-квадрат."""

import pandas as pd
from scipy.stats import chi2_contingency, chisquare, norm

ALPHA = 0.05
POWER = 0.8
SRM_ALPHA = 0.001

Z_ALPHA = norm.ppf(1 - ALPHA / 2)
Z_BETA = norm.ppf(POWER)


def check_srm(df, group_col='experiment'):
    """Проверка SRM: соответствует ли фактическое соотношение групп ожидаемому 50/50."""
    counts = df[group_col].value_counts().sort_index()
    total = counts.sum()
    expected = [total / 2, total / 2]

    _, p_value = chisquare(f_obs=counts.values, f_exp=expected)

    shares = {g: round(counts[g] / total * 100, 2) for g in counts.index}
    return shares, p_value


def compute_mde(p_pool, n_control, n_exposed):
    """Минимальный детектируемый эффект при alpha=0.05, power=0.8."""
    return (Z_ALPHA + Z_BETA) * (
        p_pool * (1 - p_pool) * (1 / n_control + 1 / n_exposed)
    ) ** 0.5


def compute_ci_diff(p_control, n_control, p_exposed, n_exposed):
    """95% доверительный интервал для разницы двух долей."""
    var_c = p_control * (1 - p_control) / n_control
    var_e = p_exposed * (1 - p_exposed) / n_exposed
    margin = Z_ALPHA * (var_c + var_e) ** 0.5

    delta = p_exposed - p_control
    return round(delta - margin, 3), round(delta + margin, 3)


def analyze_metric(df, group_col='experiment', yes_col='yes', no_col='no'):
    """Анализ метрики p = yes / (yes + no) между control и exposed."""
    sum_yes = df.groupby(group_col)[yes_col].sum()
    sum_no = df.groupby(group_col)[no_col].sum()

    n_control = sum_yes['control'] + sum_no['control']
    n_exposed = sum_yes['exposed'] + sum_no['exposed']
    p_control = sum_yes['control'] / n_control
    p_exposed = sum_yes['exposed'] / n_exposed

    p_pool = (sum_yes['control'] + sum_yes['exposed']) / (n_control + n_exposed)
    delta = p_exposed - p_control
    ci_low, ci_high = compute_ci_diff(p_control, n_control, p_exposed, n_exposed)
    mde = compute_mde(p_pool, n_control, n_exposed)

    table = pd.DataFrame(
        {'Yes': [sum_yes['control'], sum_yes['exposed']],
         'No': [sum_no['control'], sum_no['exposed']]},
        index=['control', 'exposed'],
    )
    _, p_value, _, _ = chi2_contingency(table)

    return {
        'n_control': n_control, 'n_exposed': n_exposed,
        'p_control': p_control, 'p_exposed': p_exposed,
        'delta': delta, 'ci_low': ci_low, 'ci_high': ci_high,
        'mde': mde, 'p_value': p_value,
    }