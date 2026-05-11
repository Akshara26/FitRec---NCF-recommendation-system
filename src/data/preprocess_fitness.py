from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load
df = pd.read_csv(PROJECT_ROOT / 'src/data/gym_members_exercise_tracking.csv')

# ── Step 1: Create user IDs ────
# Each row is treated as a unique user (no real user IDs in the dataset)
df['userId'] = np.arange(len(df))

# ── Step 2: Map workout types to item IDs ────
workout_types = df['Workout_Type'].unique()
workout_to_id = {w: i for i, w in enumerate(workout_types)}
df['itemId'] = df['Workout_Type'].map(workout_to_id)

print("Workout type → itemId mapping:")
print(workout_to_id)

# ── Step 3: Implicit feedback from frequency ────
# Frequency >= 4 days/week = engaged user = 1, else 0
df['rating'] = (df['Workout_Frequency (days/week)'] >= 4).astype(int)

print(f"\nImplicit feedback distribution:")
print(df['rating'].value_counts())

# ── Step 4: Keep only what NCF needs ────
fitness_df = df[['userId', 'itemId', 'rating']].copy()
fitness_df['timestamp'] = 0  # NCF expects this column, we don't have real timestamps

print(f"\nFinal shape: {fitness_df.shape}")
print(fitness_df.head())

# ── Step 5: Save ────
fitness_df.to_csv(PROJECT_ROOT / 'src/data/fitness_ratings.csv', index=False)
print("\n✅ Saved to src/data/fitness_ratings.csv")

# ── Summary ────
print("Used implicit feedback from workout frequency rather than explicit ratings.")
print(f"Users: {fitness_df.userId.nunique()}")
print(f"Items (workout types): {fitness_df.itemId.nunique()} → {list(workout_to_id.keys())}")
print(f"Positive interactions (rating=1): {df['rating'].sum()} / {len(df)}")