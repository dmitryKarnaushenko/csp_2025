import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------
# Data Loading and Aggregation
# ----------------------------
# Load the pre-cleaned dataset
data_path = 'honeyproduction_cleaned.csv'
df = pd.read_csv(data_path)


# Aggregate to national-level data by year:
df_national = df.groupby('year').agg({
    'colonies_number': 'sum',
    'production_kg': 'sum'
}).reset_index()

# For better readability, convert colonies to millions and production to millions of kilograms.
df_national['colonies_millions'] = df_national['colonies_number'] / 1e6
df_national['production_kg_millions'] = df_national['production_kg'] / 1e6

# ----------------------------
# Twin-Axis Plot: Production & Colonies Over Time
# ----------------------------
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left axis: Bee colonies (in millions) in dark blue
color_colonies = '#1f1f3f'
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Bee population (millions of colonies) during year t+1', color=color_colonies, fontsize=12)
ax1.plot(df_national['year'], df_national['colonies_millions'], marker='o', color=color_colonies, label='Bee Colonies during year t+1')
ax1.tick_params(axis='y', labelcolor=color_colonies)

# Right axis: Honey production (in millions of kg) in teal
ax2 = ax1.twinx()
color_production = '#009dcf'
ax2.set_ylabel('Honey Production (millions of kg)', color=color_production, fontsize=12)
ax2.plot(df_national['year'], df_national['production_kg_millions'], marker='o', linestyle='--', color=color_production, label='Honey Production')
ax2.tick_params(axis='y', labelcolor=color_production)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

plt.title('Honey Production and Bee Colonies Over Time', fontsize=14, pad=15)
plt.text(0.01, 0.01, "Source: USDA", transform=fig.transFigure,
         ha='left', va='bottom', fontsize=10, color='gray')
fig.tight_layout()
plt.savefig("honeyvscol", dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# Regression Analysis: Production vs. Bee Population
# ----------------------------
# We now analyze how honey production (in kg) affects bee population.
# Use aggregated (national-level) data.
# Use production_kg (in kg) as the independent variable and bee colonies as the dependent variable.
X_prod = df_national[['production_kg']].values  # production in kg
y_colonies = df_national['colonies_number'].values  # bee colonies

reg_model = LinearRegression()
reg_model.fit(X_prod, y_colonies)
y_pred = reg_model.predict(X_prod)

slope = reg_model.coef_[0]
intercept = reg_model.intercept_
r2 = reg_model.score(X_prod, y_colonies)

print("Regression Analysis: Bee Population vs Honey Production (kg)")
print(f"  Regression equation: colonies = {intercept:.2f} + {slope:.6f} * production_kg")
print(f"  R²: {r2:.4f}")

# ----------------------------
# Forecasting Population Collapse
# ----------------------------
# We want to predict when the bee population will collapse based on the regression relationship.
# Define a collapse threshold (e.g., 690,000 colonies).
collapse_threshold = 690000

# To forecast, we need to predict future production.
# Build a production (kg) vs. year model using the aggregated data.
X_year = df_national[['year']].values
y_production = df_national['production_kg'].values

prod_year_model = LinearRegression()
prod_year_model.fit(X_year, y_production)

# Forecast production from the last observed year to, say, 2200.
future_years = np.arange(df_national['year'].max() + 1, 2200).reshape(-1, 1)
future_production = prod_year_model.predict(future_years)

# Using the production vs colonies regression, predict bee population from future production values.
future_colonies = reg_model.predict(future_production.reshape(-1, 1))

# Find the first year when predicted colonies fall below the collapse threshold.
collapse_year = None
for yr, pop in zip(future_years.flatten(), future_colonies):
    if pop < collapse_threshold:
        collapse_year = yr
        break

if collapse_year:
    print(f"\nPredicted year of bee population collapse (population < {collapse_threshold} colonies): {collapse_year}")
else:
    print("\nNo population collapse predicted by 2200 based on current trends.")

# ----------------------------
# Scatter Plot: Honey Production vs. Bee Population
# ----------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_national['production_kg'], y=df_national['colonies_number'],
                color='purple', s=100, label='Data Points')

# Plot the regression line (using the same model)
x_range = np.linspace(df_national['production_kg'].min(), df_national['production_kg'].max(), 100)
y_range = reg_model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_range, color='red', linewidth=2, label='Regression Line')

plt.text(0.05, 0.95, f"R² = {r2:.4f}", transform=plt.gca().transAxes,
         ha='left', va='top', fontsize=12, color='black', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.title('Regression: Bee Population during year t+1 vs Honey Production (kg)', fontsize=14)
plt.xlabel('Total Honey Production (kg)')
plt.ylabel('Total Bee Colonies during year t+1')
plt.legend(loc='upper right')
plt.grid(True)
plt.text(0.01, 0.01, "Source: USDA", transform=plt.gcf().transFigure,
         ha='left', va='bottom', fontsize=10, color='gray')
plt.tight_layout()
plt.savefig("regression.png", dpi=300, bbox_inches='tight')
plt.show()
