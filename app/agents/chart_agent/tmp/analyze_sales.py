import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=" * 60)
print("SALES DATA ANALYSIS REPORT")
print("=" * 60)

# Basic overview
print("\n1. DATA OVERVIEW")
print("-" * 40)
print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"   Number of records: {len(df)}")
print(f"   Products: {df['Product'].nunique()}")
print(f"   Columns: {', '.join(df.columns)}")

# Revenue summary
print("\n2. OVERALL REVENUE SUMMARY")
print("-" * 40)
print(f"   Total Revenue:   ${df['Revenue'].sum():,.2f}")
print(f"   Total Units Sold:{df['Units Sold'].sum():,}")
print(f"   Avg Revenue/Day: ${df['Revenue'].sum()/df['Date'].nunique():,.2f}")
print(f"   Avg Units/Day:   {df['Units Sold'].sum()/df['Date'].nunique():,.2f}")

# Per-product breakdown
print("\n3. PER-PRODUCT SUMMARY")
print("-" * 40)
product_stats = df.groupby('Product').agg(
    Total_Units=('Units Sold', 'sum'),
    Total_Revenue=('Revenue', 'sum'),
    Avg_Entry=('Revenue', 'mean'),
    Days_Active=('Date', 'nunique')
).reset_index()
for _, row in product_stats.iterrows():
    print(f"\n   {row['Product']}:")
    print(f"     Total Units:   {int(row['Total_Units'])}")
    print(f"     Total Revenue: ${row['Total_Revenue']:,.2f}")
    print(f"     Avg Revenue/Entry: ${row['Avg_Entry']:,.2f}")
    print(f"     Active Days:   {int(row['Days_Active'])}")

# Revenue share
print("\n4. REVENUE SHARE BY PRODUCT (%)")
print("-" * 40)
total_rev = df['Revenue'].sum()
for product in sorted(df['Product'].unique()):
    share = df[df['Product'] == product]['Revenue'].sum() / total_rev * 100
    print(f"   {product}: {share:5.1f}%")

# Daily trends
print("\n5. DAILY TRENDS")
print("-" * 40)
daily = df.groupby('Date').agg(
    Units=('Units Sold', 'sum'),
    Revenue=('Revenue', 'sum')
).reset_index()
for _, row in daily.iterrows():
    print(f"   {row['Date'].strftime('%Y-%m-%d')}  |  Units: {int(row['Units']):>3}  |  Revenue: ${row['Revenue']:>5,.2f}")

print(f"\n   Peak day:     {daily.loc[daily['Revenue'].idxmax(), 'Date'].strftime('%Y-%m-%d')} (${daily['Revenue'].max():,.2f})")
print(f"   Lowest day:   {daily.loc[daily['Revenue'].idxmin(), 'Date'].strftime('%Y-%m-%d')} (${daily['Revenue'].min():,.2f})")

# Trend direction
x = np.arange(len(daily))
slope_rev, intercept_rev, r_rev, p_rev, _ = stats.linregress(x, daily['Revenue'])
slope_units, _, r_units, p_units, _ = stats.linregress(x, daily['Units'])

print("\n6. TREND ANALYSIS (Linear Regression)")
print("-" * 40)
print(f"   Revenue trend:   {slope_rev:+.2f}/day  (R²={r_rev**2:.4f}, p={p_rev:.4f})")
print(f"   Units trend:     {slope_units:+.2f}/day  (R²={r_units**2:.4f}, p={p_units:.4f})")
if r_rev**2 > 0.5:
    direction = "upward" if slope_rev > 0 else "downward"
    strength = "strong " if r_rev**2 > 0.8 else ""
    print(f"   => Revenue shows a {strength}{direction} trend (R²={r_rev**2:.4f})")
else:
    print(f"   => Revenue trend is {'weak or inconsistent' if r_rev**2 < 0.2 else 'moderate'} (R²={r_rev**2:.4f})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
