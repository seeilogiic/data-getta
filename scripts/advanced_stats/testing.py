import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
import re
import json
import numpy as np
from typing import Dict, Tuple, List, Set
from pathlib import Path
import json

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / '.env')

# Supabase configuration
SUPABASE_URL = os.getenv("VITE_SUPABASE_PROJECT_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_PROJECT_URL and SUPABASE_API_KEY must be set in .env file"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_table_to_json(supabase: Client, table_name: str, output_file: str, batch_size: int = 200):
    offset = 0
    first_batch = True

    with open(output_file, "w") as f:
        f.write("[")  # Start JSON array

        while True:
            # Fetch rows in the range [offset, offset+batch_size-1]
            response = (
                supabase.table(table_name)
                .select("*")
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            data = response.data

            if not data:  # Stop when no more rows
                break

            # Write batch to file
            for row in data:
                if not first_batch:
                    f.write(",\n")  # Add comma between records
                json.dump(row, f, indent=4)
                first_batch = False

            offset += batch_size

        f.write("]")  # Close JSON array

    print(f"Data saved to {output_file} ✅")

fetch_table_to_json(supabase, "BattedBalls", "output.json", batch_size=200)
