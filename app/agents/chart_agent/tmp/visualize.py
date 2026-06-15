import os
import sys

# Check packages
for pkg in ['pandas', 'matplotlib', 'seaborn']:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        os.system(f'python -m pip install {pkg}')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d')
requires_scaling = False

# --- Chart 1: Daily Revenue Trend ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#5B9BD5', '#ED7D31', '#A5A5A5']
bar_width = 0.35
x = np.arange(len(df))

ax.bar(x, df['Units Sold'], bar_width, label='Units Sold', color='#5B9BD5')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Daily Units Sold', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(df['Date'], rotation=45)
for i, v in enumerate(df['Units Sold']):
    ax.text(i, v + 0.3, str(v), ha='center', fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig('/units_sold.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved /units_sold.png")

# --- Chart 2: Daily Revenue Trend ---
fig, ax = plt.subplots(figsize=(10, 6))
slopes = []
for label in sorted(df[['Date','Revenue']].values):
    pass
ax.plot(df['Date'], df['Revenue'], linewidth=2, color='#ED7D31', marker='o', markersize=8, label='Revenue')
ax.fill_between(x, df['Revenue'], alpha=0.15, color='#ED7D31')
for i, v in enumerate(df['Revenue']):
    ax.annotate(f'${v}', (x[i], v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Revenue ($)', fontsize=12)
ax.set_title('Daily Revenue Trend', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(df['Date'], rotation=45)
for fac in ['y']:
    ax.grid(axis=fac, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig('/revenue_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved /revenue_trend.png")

# --- Chart 3: Product Performance ---
product_stats = df.groupby('Product').agg(
    Total_Units=('Units Sold', 'sum'),
    Total_Revenue=('Revenue', 'sum')
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
n = len(product_stats)
x_p = np.arange(n)
width = 0.35

bars_u = ax.bar(x_p - width/2, product_stats['Total_Units'], width, label='Units Sold', color='#5B9BD5')
bars_r = ax.bar(x_p + width/2, product_stats['Total_Revenue'], width, label='Revenue ($)', color='#ED7D31')

for bar in bars_u:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
for bar in bars_r:
    height = bar.get_height()
    ax.annotate(f'${int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

ax.set_xlabel('Product', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Product Performance Overview', fontsize=16)
ax.set_xticks(x_p)
ax.set_xticklabels(product_stats['Product'])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/product_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved /product_comparison.png")

print("All charts saved to / directory.")
