import os
import re
import csv

def extract_from_dir(base_dir):
    results = {}
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return results
        
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirs:
        # Match seed number at the end of the folder name (e.g. seed001 or seed010)
        match = re.search(r"seed0*(\d+)$", subdir)
        if match:
            seed_num = int(match.group(1))
            summary_path = os.path.join(base_dir, subdir, "sample_runs_summary.csv")
            
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row["run"].strip() == "mean":
                                tardiness = float(row["total_tardiness"])
                                releases = float(row["release_count"])
                                results[seed_num] = f"{int(round(tardiness))} / {int(round(releases))}"
                                break
                except Exception as e:
                    print(f"Error reading {summary_path}: {e}")
            else:
                print(f"Summary file not found: {summary_path}")
                
    return results

def main():
    dir_8env = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\換題差異\8env"
    dir_Rstab = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\換題差異\Rstab"
    
    results_8env = extract_from_dir(dir_8env)
    results_Rstab = extract_from_dir(dir_Rstab)
    
    print("\n================ Extracted PPO Results ================")
    print(f"| Seed | 8env (Tardiness / Releases) | Rstab (Tardiness / Releases) |")
    print(f"| :--- | :--- | :--- |")
    for s in range(1, 11):
        r_8 = results_8env.get(s, "N/A")
        r_R = results_Rstab.get(s, "N/A")
        print(f"| {s} | {r_8} | {r_R} |")

if __name__ == "__main__":
    main()
