import pandas as pd
import numpy as np
from scipy import stats

DATA_FILE = "Cleaned_Sales_Dataset.xlsx"
ALPHA = 0.05  # significance level


def load_data(path):
    df = pd.read_excel(path)
    print(f"Loaded {len(df):,} orders spanning {df['Order_Date'].min().date()} "
          f"to {df['Order_Date'].max().date()}\n")
    return df


def business_recap(df):
    """Quick recap of the headline numbers from Parts 1-3 (EDA)."""
    print("=" * 60)
    print("BUSINESS RECAP (Parts 1-3)")
    print("=" * 60)

    total_revenue = df["Total_Sales"].sum()
    print(f"Total revenue............ ₹{total_revenue:,.0f}")
    print(f"Total orders.............. {len(df):,}")
    print(f"Average order value....... ₹{df['Total_Sales'].mean():,.0f}")

    top_category = df.groupby("Category")["Total_Sales"].sum().idxmax()
    top_city = df.groupby("City")["Total_Sales"].sum().idxmax()
    top_product = df.groupby("Product")["Total_Sales"].sum().idxmax()
    print(f"Leading category.......... {top_category}")
    print(f"Leading city............... {top_city}")
    print(f"Best-selling product....... {top_product}\n")


def cohens_d(a, b):
    """Pooled-variance effect size for two independent samples."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_std


def hypothesis_1_weekday_vs_weekend(df):
    """
    H0: Average order value on weekends equals average order value on weekdays.
    H1: Average order value on weekends differs from average order value on weekdays.
    Test: Independent-samples two-tailed T-test.
    """
    print("=" * 60)
    print("HYPOTHESIS 1 — Weekday vs Weekend Order Value (T-Test)")
    print("=" * 60)
    print('Business hypothesis: "Orders placed on weekends have a')
    print(' statistically significant different value than orders')
    print(' placed on weekdays."\n')

    df["Day_Type"] = np.where(df["Order_Weekday"].isin(["Saturday", "Sunday"]),
                               "Weekend", "Weekday")
    weekend = df.loc[df["Day_Type"] == "Weekend", "Total_Sales"]
    weekday = df.loc[df["Day_Type"] == "Weekday", "Total_Sales"]

    print(f"Weekday orders: n={len(weekday)}, mean=₹{weekday.mean():,.0f}, "
          f"std=₹{weekday.std():,.0f}")
    print(f"Weekend orders: n={len(weekend)}, mean=₹{weekend.mean():,.0f}, "
          f"std=₹{weekend.std():,.0f}")

    # Check the equal-variance assumption before picking the T-test variant
    levene_stat, levene_p = stats.levene(weekend, weekday)
    equal_var = levene_p > ALPHA
    print(f"\nLevene's test for equal variances: p={levene_p:.4f} "
          f"({'variances equal -> Student t-test' if equal_var else 'variances unequal -> Welch t-test'})")

    t_stat, p_value = stats.ttest_ind(weekend, weekday, equal_var=equal_var)

    diff = weekend.mean() - weekday.mean()
    se = np.sqrt(weekend.var(ddof=1) / len(weekend) + weekday.var(ddof=1) / len(weekday))
    ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
    d = cohens_d(weekend, weekday)

    print(f"\nt-statistic = {t_stat:.3f}")
    print(f"p-value     = {p_value:.4f}")
    print(f"Mean difference (Weekend - Weekday) = ₹{diff:,.0f}")
    print(f"95% CI of the difference = (₹{ci_low:,.0f}, ₹{ci_high:,.0f})")
    print(f"Cohen's d (effect size) = {d:.3f}")

    if p_value < ALPHA:
        print(f"\nResult: p < {ALPHA} -> reject H0. The difference is statistically significant.")
    else:
        print(f"\nResult: p >= {ALPHA} -> fail to reject H0. No statistically significant "
              "difference was found.")
    print("Business conclusion: order value does not meaningfully depend on whether the "
          "order was placed on a weekday or a weekend. Marketing spend timed around the "
          "weekend is unlikely to change basket size on its own.\n")

    return {"test": "T-test", "p_value": p_value, "t_stat": t_stat, "diff": diff,
            "ci": (ci_low, ci_high), "effect_size": d}


def hypothesis_2_category_vs_gender(df):
    """
    H0: Product category chosen is independent of customer gender.
    H1: Product category chosen is associated with customer gender.
    Test: Chi-squared test of independence on a Gender x Category contingency table.
    """
    print("=" * 60)
    print("HYPOTHESIS 2 — Category Preference vs Gender (Chi-Squared Test)")
    print("=" * 60)
    print('Business hypothesis: "The category customers buy from is')
    print(' associated with their gender."\n')

    contingency = pd.crosstab(df["Gender"], df["Category"])
    print("Observed order counts:")
    print(contingency, "\n")

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    print(f"Chi-squared statistic = {chi2:.3f}")
    print(f"Degrees of freedom    = {dof}")
    print(f"p-value               = {p_value:.4f}")
    print(f"Cramer's V (effect size) = {cramers_v:.3f}  (0=no association, 1=perfect association)")

    if p_value < ALPHA:
        print(f"\nResult: p < {ALPHA} -> reject H0. Category choice is significantly "
              "associated with gender.")
    else:
        print(f"\nResult: p >= {ALPHA} -> fail to reject H0. No statistically significant "
              "association was found.")
    print("Business conclusion: category-level marketing does not need to be split by "
          "gender based on this sample — men and women are buying across categories in "
          "roughly the proportions we'd expect if gender played no role.\n")

    return {"test": "Chi-squared", "p_value": p_value, "chi2": chi2, "dof": dof,
            "effect_size": cramers_v}


def main():
    df = load_data(DATA_FILE)
    business_recap(df)
    h1 = hypothesis_1_weekday_vs_weekend(df)
    h2 = hypothesis_2_category_vs_gender(df)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"H1 (Weekday vs Weekend spend) : p={h1['p_value']:.4f} -> "
          f"{'SIGNIFICANT' if h1['p_value'] < ALPHA else 'NOT significant'}")
    print(f"H2 (Category vs Gender)       : p={h2['p_value']:.4f} -> "
          f"{'SIGNIFICANT' if h2['p_value'] < ALPHA else 'NOT significant'}")


if __name__ == "__main__":
    main()
