import os
import csv

scenarios = {
    "多單": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\多單",
    "瓶頸": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\瓶頸",
    "普通": r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260701\普通"
}

def transpose_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    # Read original content
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    if not rows:
        print(f"Empty CSV: {csv_path}")
        return
        
    # Column keys in transposed CSV
    # First column: Seed
    # Other columns: Policy_Metric
    col_names = []
    for r in rows:
        col_name = f"{r['Policy']}_{r['Metric']}"
        col_names.append(col_name)
        
    new_headers = ["Seed"] + col_names
    
    # We want 12 rows in the new CSV: seed1..seed10, Mean, Std
    seed_keys = [f"seed{i}" for i in range(1, 11)] + ["Mean", "Std"]
    
    transposed_rows = []
    for s_key in seed_keys:
        new_row = {"Seed": s_key}
        for r in rows:
            col_name = f"{r['Policy']}_{r['Metric']}"
            new_row[col_name] = r[s_key]
        transposed_rows.append(new_row)
        
    # Overwrite the original file with transposed data
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        for tr in transposed_rows:
            writer.writerow(tr)
            
    print(f"Successfully transposed: {csv_path}")

def main():
    for sc_name, sc_path in scenarios.items():
        csv_path = os.path.join(sc_path, "eval_baseline_10seeds_latest_comparison.csv")
        transpose_csv(csv_path)

if __name__ == "__main__":
    main()
