import sys
import os
import pandas as pd

language = sys.argv[1]
split = sys.argv[2]

base = f"/home/siplabiith/ind_lang_trials_v2/output/{language}"
input_path = f"{base}/trials_{split}_final.csv"
output_path = f"{base}/trials_{split}_crossview.csv"

if not os.path.exists(input_path):
    print(f"SKIP {language}/{split} — file not found: {input_path}")
    sys.exit(0)

df = pd.read_csv(input_path)

cross_view_trials = df[["file1", "word1", "word2", "gt_decision"]].drop_duplicates(
    subset=["file1", "word2"]
)

cross_view_trials.to_csv(output_path, index=False)

print(f"{language}/{split}: {len(df):,} -> {len(cross_view_trials):,} rows")
