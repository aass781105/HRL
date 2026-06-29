import os
import re
import csv

base_dir = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260624\cad1_greedy"
output_csv_path = os.path.join(base_dir, "global_total_tardiness_comparison.csv")

def extract_seed_num(folder_name):
    # E.g. 20260622_153706_odprog_seed001 -> 1
    m = re.search(r"seed0*(\d+)$", folder_name)
    if m:
        return int(m.group(1))
    return None

def main():
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
        
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    seed_folders = {}
    for f in folders:
        seed = extract_seed_num(f)
        if seed is not None and 1 <= seed <= 10:
            seed_folders[seed] = f
            
    print(f"Mapped {len(seed_folders)} seeds to their folders:")
    for s in sorted(seed_folders.keys()):
        print(f"  Seed {s} -> {seed_folders[s]}")
        
    # We will gather data as: event_id -> {seed_num -> value}
    data_matrix = {}
    
    for s in range(1, 11):
        if s not in seed_folders:
            print(f"Warning: Seed {s} folder not found!")
            continue
            
        folder_path = os.path.join(base_dir, seed_folders[s])
        
        # Find release log file (e.g. *release_log.csv)
        log_files = [f for f in os.listdir(folder_path) if "release_log" in f and f.endswith(".csv")]
        if not log_files:
            print(f"Error: No release log CSV found in {folder_path}!")
            continue
            
        log_file_path = os.path.join(folder_path, log_files[0])
        print(f"Reading Seed {s} release log: {log_files[0]}")
        
        with open(log_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_type = row.get("Release_Type", "").strip()
                # Skip SUMMARY row
                if event_type == "SUMMARY" or not row.get("Event_ID"):
                    continue
                    
                event_id = int(row["Event_ID"])
                tardiness_val = float(row.get("Global_Total_Tardiness", 0.0))
                
                if event_id not in data_matrix:
                    data_matrix[event_id] = {}
                data_matrix[event_id][s] = tardiness_val

    # Write merged table to CSV
    sorted_events = sorted(data_matrix.keys())
    headers = ["Event_ID"] + [f"Seed{s}" for s in range(1, 11)]
    
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for event_id in sorted_events:
            row = [event_id]
            for s in range(1, 11):
                val = data_matrix[event_id].get(s, "")
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            writer.writerow(row)
            
    print(f"\nSuccessfully generated merged CSV at: {output_csv_path}")

if __name__ == "__main__":
    main()
