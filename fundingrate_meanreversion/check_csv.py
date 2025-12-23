import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), "funding_rates.csv")
df = pd.read_csv(csv_path)
print("Start:", df['timestamp'].min())
print("End:", df['timestamp'].max())
print("Count:", len(df))
