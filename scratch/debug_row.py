import csv
import os

def main():
    target_file = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\PPO瓶頸\20260706_150950_odprog_seed001\odprog_raw_state.csv"
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return
        
    with open(target_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                print(f"  {k}: {repr(v)}")
            break

if __name__ == "__main__":
    main()
