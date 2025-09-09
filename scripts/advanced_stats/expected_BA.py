import os
import pandas as pd
import torch
import joblib
from supabase import create_client, Client
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv
from collections import Counter

# ----------------------------
# Project root and environment
# ----------------------------
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

SUPABASE_URL = os.getenv("VITE_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("VITE_SUPABASE_PROJECT_URL and VITE_SUPABASE_API_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Define PyTorch model (same as training)
# ----------------------------
class HitPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# Paths for scaler and models
# ----------------------------
scaler_path = project_root / "training_models" / "models" / "scaler.save"
if not scaler_path.exists():
    raise FileNotFoundError("Scaler file not found. Run training script first.")
scaler = joblib.load(scaler_path)

model_dir = project_root / "training_models" / "models"
model_files = sorted(model_dir.glob("xba_model_*.pt"))
if not model_files:
    raise FileNotFoundError("No trained model files found in the models directory.")
MODEL_PATH = model_files[-1]
print(f"Latest model found: {MODEL_PATH}")

# ----------------------------
# Load trained model
# ----------------------------
features = ['exit_speed', 'launch_angle', 'direction', 'hit_spin_rate', 'distance', 'bearing', 'hang_time']
model = HitPredictor(len(features))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ----------------------------
# Helper function: fetch table in batches
# ----------------------------
def fetch_table_batched(table_name, batch_size=1000, select="*"):
    offset = 0
    df_list = []
    while True:
        response = (
            supabase.table(table_name)
            .select(select)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        data = response.data
        if not data:
            break
        df_list.append(pd.DataFrame(data))
        offset += batch_size
        print(f"Fetched {len(data)} rows from {table_name}, batch {offset // batch_size}")
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

# ----------------------------
# Fetch BattedBalls in batches
# ----------------------------
def fetch_batted_balls(batch_size=1000):
    offset = 0
    while True:
        response = (
            supabase.table("BattedBalls")
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        data = response.data
        if not data:
            break
        offset += batch_size
        print(f"Fetched BattedBalls batch {offset // batch_size}: {len(data)} records")
        yield pd.DataFrame(data)

# ----------------------------
# Process BattedBalls batches
# ----------------------------
print("Fetching BattedBalls from Supabase in batches...")
df_list = []
for df_batch in fetch_batted_balls(batch_size=1000):
    if df_batch.empty:
        continue
    df_batch = df_batch.dropna(subset=features)
    df_batch = df_batch[df_batch['exit_speed'].astype(float) > 0]

    if df_batch.empty:
        continue

    X_bb = df_batch[features].astype(float).values
    X_scaled = scaler.transform(X_bb)

    with torch.no_grad():
        df_batch['xBA_prob'] = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy().flatten()

    df_list.append(df_batch)

if not df_list:
    print("No valid batted balls to process.")
    exit()

df_bb = pd.concat(df_list, ignore_index=True)

# ----------------------------
# Collect batter_ids from BattedBalls as strings
# ----------------------------
batted_ball_ids = set(df_bb['batter_id'].dropna().astype(str).tolist())

# ----------------------------
# Sum expected hits per batter_id
# ----------------------------
batter_xba = df_bb.groupby('batter_id').agg(
    expected_hits=('xBA_prob', 'sum')
).reset_index()

# ----------------------------
# Fetch Players in batches and filter by BattedBalls batter_ids
# ----------------------------
def fetch_players_filtered(batch_size=1000):
    offset = 0
    filtered_players = []
    while True:
        response = (
            supabase.table("Players")
            .select("BatterId, Name, TeamTrackmanAbbreviation, Year")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        data = response.data
        if not data:
            break

        df_batch = pd.DataFrame(data)
        before_count = len(df_batch)
        df_batch = df_batch[df_batch['BatterId'].notna() & df_batch['BatterId'].astype(str).isin(batted_ball_ids)]
        skipped = before_count - len(df_batch)
        print(f"Fetched Players batch {offset // batch_size + 1}: {before_count} rows, skipped {skipped} players")

        filtered_players.append(df_batch)
        offset += batch_size

    if filtered_players:
        return pd.concat(filtered_players, ignore_index=True)
    else:
        return pd.DataFrame()

df_players = fetch_players_filtered(batch_size=1000)

# ----------------------------
# Fetch BatterStats
# ----------------------------
df_batterstats = fetch_table_batched("DevBatterStats", select="Batter, BatterTeam, Year, at_bats")

# ----------------------------
# Clean string columns for safe merge
# ----------------------------
for col in ['Name', 'TeamTrackmanAbbreviation']:
    if col in df_players.columns:
        df_players[col] = df_players[col].astype(str).str.strip().str.upper()
for col in ['Batter', 'BatterTeam']:
    if col in df_batterstats.columns:
        df_batterstats[col] = df_batterstats[col].astype(str).str.strip().str.upper()

# ----------------------------
# Compute total_batted_balls per batter using all Players entries
# ----------------------------
total_batted_balls_map = {}

for b_id, group in df_players.groupby('BatterId'):
    name = group['Name'].iloc[0]
    year = group['Year'].iloc[0]

    # Find all DevBatterStats rows matching Name + Year + any Team in group
    teams = group['TeamTrackmanAbbreviation'].tolist()
    matching_stats = df_batterstats[
        (df_batterstats['Batter'] == name) &
        (df_batterstats['Year'] == year) &
        (df_batterstats['BatterTeam'].isin(teams))
    ]
    total_at_bats = matching_stats['at_bats'].sum()
    total_batted_balls_map[b_id] = total_at_bats

# ----------------------------
# Join expected hits -> Players -> BatterStats
# ----------------------------
df_join = batter_xba.merge(df_players, left_on='batter_id', right_on='BatterId', how='left')

# Assign total_batted_balls from sum above
df_join['total_batted_balls'] = df_join['BatterId'].map(total_batted_balls_map).fillna(0).astype(int)

# ----------------------------
# Compute expected batting average (xBA)
# ----------------------------
df_join['expected_batting_average'] = (
    (df_join['expected_hits'] / df_join['total_batted_balls'])
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
    .round(3)
)

# ----------------------------
# Prepare data for upsert
# ----------------------------
batter_records = []
for _, row in df_join.iterrows():
    batter_records.append({
        "batter_id": row['batter_id'],
        "expected_hits": float(row['expected_hits']),
        "expected_batting_average": float(row['expected_batting_average']),
        "total_batted_balls": int(row['total_batted_balls']),
        "updated_at": datetime.now(timezone.utc).isoformat()
    })

# ----------------------------
# Debug step: check for duplicate batter_ids before upsert
# ----------------------------
id_counts = Counter([rec['batter_id'] for rec in batter_records])
duplicates = {b_id: count for b_id, count in id_counts.items() if count > 1}
if duplicates:
    print("⚠️ Duplicate batter_ids found in upsert batch:")
    for b_id, count in duplicates.items():
        print(f"batter_id={b_id}, count={count}")
else:
    print("No duplicate batter_ids in upsert batch.")

# ----------------------------
# Deduplicate records by batter_id, keep last
# ----------------------------
df_upsert = pd.DataFrame(batter_records)
df_upsert = df_upsert.drop_duplicates(subset='batter_id', keep='last')
batter_records = df_upsert.to_dict(orient='records')

# ----------------------------
# Upsert into AdvancedBatterStats
# ----------------------------
print(f"Upserting {len(batter_records)} records into AdvancedBatterStats...")

try:
    batch_size = 100
    total_upserted = 0
    for i in range(0, len(batter_records), batch_size):
        batch = batter_records[i:i + batch_size]
        supabase.table("AdvancedBatterStats").upsert(batch, on_conflict="batter_id").execute()
        total_upserted += len(batch)
        print(f"Upserted batch {i//batch_size + 1}: {len(batch)} records")
    print(f"Successfully upserted {total_upserted} batter records")
except Exception as e:
    print(f"Error upserting AdvancedBatterStats: {e}")
