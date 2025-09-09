import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from collections import defaultdict

# ------------------------------
# Load environment variables
# ------------------------------
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

SUPABASE_URL = os.getenv("VITE_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("VITE_SUPABASE_PROJECT_URL and VITE_SUPABASE_API_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# Helper Functions
# ------------------------------
def fetch_batted_balls_in_batches(batch_size: int = 500):
    all_data = []
    offset = 0

    while True:
        response = (
            supabase.table("BattedBalls")
            .select("batter_id, exit_speed")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        data = response.data

        if not data:
            break

        all_data.extend(data)
        offset += batch_size

        print(f"Fetched batch {offset // batch_size}: {len(data)} records")

    return all_data


def calculate_avg_exit_velocity(batted_balls):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for row in batted_balls:
        batter_id = row.get("batter_id")
        exit_speed = row.get("exit_speed")

        # Exclude nulls and 0 (tracker failed)
        if batter_id is not None and exit_speed is not None:
            try:
                exit_speed_val = float(exit_speed)
                if exit_speed_val > 0:  # ignore failed readings
                    totals[batter_id] += exit_speed_val
                    counts[batter_id] += 1
            except ValueError:
                continue

    avg_results = []
    for batter_id in totals:
        avg_exit_velocity = totals[batter_id] / counts[batter_id]
        avg_results.append({
            "batter_id": batter_id,
            "avg_exit_velocity": avg_exit_velocity
        })

    return avg_results


def upload_avg_exit_velocity(avg_results):
    if not avg_results:
        print("No averages to upload")
        return

    try:
        print(f"Preparing to upload {len(avg_results)} average exit velocities...")

        batch_size = 100
        total_inserted = 0

        for i in range(0, len(avg_results), batch_size):
            batch = avg_results[i : i + batch_size]

            try:
                supabase.table("AdvancedBatterStats").upsert(
                    batch, on_conflict="batter_id"
                ).execute()
                total_inserted += len(batch)
                print(f"Uploaded batch {i//batch_size + 1}: {len(batch)} records")
            except Exception as batch_error:
                print(f"Error uploading batch {i//batch_size + 1}: {batch_error}")
                if batch:
                    print(f"Sample failed record: {batch[0]}")

        print(f"Successfully processed {total_inserted} AdvancedBatterStats records")

    except Exception as e:
        print(f"Supabase upload error: {e}")


def main():
    print("=== Starting average exit velocity processing ===")

    batted_balls = fetch_batted_balls_in_batches(batch_size=200)
    print(f"\nTotal batted balls fetched: {len(batted_balls)}")

    avg_results = calculate_avg_exit_velocity(batted_balls)
    print(f"Calculated averages for {len(avg_results)} batters")

    if avg_results:
        print("\nUploading averages to Supabase...")
        upload_avg_exit_velocity(avg_results)
    else:
        print("No averages calculated to upload")


if __name__ == "__main__":
    main()
