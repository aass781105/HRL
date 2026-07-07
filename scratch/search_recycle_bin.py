import os
import shutil

def main():
    recycle_bin = "C:\\$Recycle.Bin"
    if not os.path.exists(recycle_bin):
        print(f"Recycle bin path not found: {recycle_bin}")
        return
        
    print(f"Scanning Recycle Bin: {recycle_bin}...")
    found_runs = []
    
    # We walk through the Recycle Bin
    for root, dirs, files in os.walk(recycle_bin):
        # We look for folders containing sample_runs_summary.csv
        if "sample_runs_summary.csv" in files:
            summary_path = os.path.join(root, "sample_runs_summary.csv")
            # Let's inspect the first row of this summary to see if it is our 300 horizon runs
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Check if it has the timestamp we are looking for (20260706_1758 or 20260706_1805)
                # or check if it matches 300 horizon features
                found_runs.append({
                    "path": root,
                    "summary_path": summary_path,
                    "content_preview": content[:200]
                })
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")
                
    print(f"\nFound {len(found_runs)} candidate runs in Recycle Bin:")
    for idx, run in enumerate(found_runs):
        print(f"\nCandidate [{idx}]:")
        print(f"  Path in Bin: {run['path']}")
        print(f"  Preview: {run['content_preview'].strip()}")
        
if __name__ == "__main__":
    main()
