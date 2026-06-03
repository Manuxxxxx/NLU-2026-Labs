import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

log_dir = "./runs"
all_data = []

print("Scanning runs directory...")

# 1. Traverse all folders and find event files inside 'tb/' subdirectories
for run_name in os.listdir(log_dir):
    run_path = os.path.join(log_dir, run_name)
    if not os.path.isdir(run_path):
        continue
    
    # Target the 'tb' subfolder where PyTorch put the files
    tb_path = os.path.join(run_path, "tb")
    target_path = tb_path if os.path.isdir(tb_path) else run_path
    
    # Determine the group prefix (1A, 1B, 2A, 2B)
    prefix = run_name[:2]
    if prefix not in ["1A", "1B", "2A", "2B"]:
        continue  # Skip any folders that don't match your schema
        
    try:
        ea = event_accumulator.EventAccumulator(target_path)
        ea.Reload()
        
        # Pull all metrics out
        for tag in ea.Tags()['scalars']:
            scalars = ea.Scalars(tag)
            for s in scalars:
                all_data.append({
                    'Group': prefix,
                    'Run': run_name,
                    'Metric': tag,
                    'Step': s.step,
                    'Value': s.value
                })
        print(f" Loaded [{prefix}]: {run_name}")
    except Exception as e:
        print(f"❌ Failed reading {run_name}: {e}")

# 2. Split data into your 6 specific report views
if all_data:
    df = pd.DataFrame(all_data)
    
    # Define filters
    filters = {
        "only_1A": df[df['Group'] == '1A'],
        "only_1B": df[df['Group'] == '1B'],
        "1A_plus_1B": df[df['Group'].isin(['1A', '1B'])],
        "only_2A": df[df['Group'] == '2A'],
        "only_2B": df[df['Group'] == '2B'],
        "2A_plus_2B": df[df['Group'].isin(['2A', '2B'])]
    }
    
    print("\n--- Exporting Files ---")
    for name, filtered_df in filters.items():
        if not filtered_df.empty:
            filename = f"export_{name}.csv"
            filtered_df.to_csv(filename, index=False)
            print(f" Saved: {filename} ({len(filtered_df)} rows)")
        else:
            print(f"⚠️ No data found for configuration: {name}")
            
    print("\n All done! You have 6 individual CSVs ready for your LaTeX report graphs.")
else:
    print("\n❌ Error: No scalar data extracted. Double check folder names and event locations.")