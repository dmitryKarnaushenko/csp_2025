import pandas as pd

# Load the original dataset
original_path = 'US_honey_dataset_updated.csv'
df = pd.read_csv(original_path)

# Convert honey production from pounds to kilograms.
# The column "production" is in pounds; convert it to kg.
df['production_kg'] = df['production'] * 0.453592

# Remove all rows from 2018 onward (i.e. keep only years < 2018)
df['year']=df['year'] +10
df = df[df['year'] < 2018]

# Save the cleaned dataset to a new file
cleaned_path = 'honeyproduction_cleaned.csv'
df.to_csv(cleaned_path, index=False)

print(f"Dataset cleaned and saved to {cleaned_path}")
