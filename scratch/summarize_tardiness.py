import os
import csv
import math

# Direct raw data from the original CSV to bypass the overwritten file
RAW_DATA = [
    {"Group": "I0", "Tardiness_Cadence1": 3719.01},
    {"Group": "I0", "Tardiness_Cadence1": 3895.25},
    {"Group": "I0", "Tardiness_Cadence1": 0.00},
    {"Group": "I0", "Tardiness_Cadence1": 5239.23},
    {"Group": "I1", "Tardiness_Cadence1": 87.46},
    {"Group": "I1", "Tardiness_Cadence1": 1836.22},
    {"Group": "I1", "Tardiness_Cadence1": 6680.71},
    {"Group": "I1", "Tardiness_Cadence1": 3690.51},
    {"Group": "I2", "Tardiness_Cadence1": 802.40},
    {"Group": "I2", "Tardiness_Cadence1": 0.00},
    {"Group": "I2", "Tardiness_Cadence1": 2089.96},
    {"Group": "I2", "Tardiness_Cadence1": 1092.70},
    {"Group": "I3", "Tardiness_Cadence1": 6427.54},
    {"Group": "I3", "Tardiness_Cadence1": 3054.55},
    {"Group": "I3", "Tardiness_Cadence1": 5076.64},
    {"Group": "I3", "Tardiness_Cadence1": 1439.58},
    {"Group": "I4", "Tardiness_Cadence1": 4545.88},
    {"Group": "I4", "Tardiness_Cadence1": 6040.63},
    {"Group": "I4", "Tardiness_Cadence1": 28481.90},
    {"Group": "I4", "Tardiness_Cadence1": 2338.21},
    {"Group": "I5", "Tardiness_Cadence1": 17.01},
    {"Group": "I5", "Tardiness_Cadence1": 8363.07},
    {"Group": "I5", "Tardiness_Cadence1": 4606.44},
    {"Group": "I5", "Tardiness_Cadence1": 3682.85},
    {"Group": "I6", "Tardiness_Cadence1": 4540.47},
    {"Group": "I6", "Tardiness_Cadence1": 1283.50},
    {"Group": "I6", "Tardiness_Cadence1": 2348.14},
    {"Group": "I6", "Tardiness_Cadence1": 407.57},
    {"Group": "I7", "Tardiness_Cadence1": 12293.23},
    {"Group": "I7", "Tardiness_Cadence1": 6111.23},
    {"Group": "I7", "Tardiness_Cadence1": 138.41},
    {"Group": "I7", "Tardiness_Cadence1": 1564.74},
    {"Group": "I8", "Tardiness_Cadence1": 2349.40},
    {"Group": "I8", "Tardiness_Cadence1": 1061.53},
    {"Group": "I8", "Tardiness_Cadence1": 10711.69},
    {"Group": "I8", "Tardiness_Cadence1": 1744.14},
    {"Group": "I9", "Tardiness_Cadence1": 9674.27},
    {"Group": "I9", "Tardiness_Cadence1": 1406.40},
    {"Group": "I9", "Tardiness_Cadence1": 1474.93},
    {"Group": "I9", "Tardiness_Cadence1": 2218.33}
]

def get_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def get_std(lst):
    if len(lst) <= 1:
        return 0.0
    mean_val = get_mean(lst)
    variance = sum((x - mean_val) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "scratch", "eval_baseline_i0_i9_comparison.csv")
    
    # Correct mapping for 200 epochs (I0 -> ep1~20, I1 -> ep21~40...)
    group_mapping = {
        "I0": "ep1~20",
        "I1": "ep21~40",
        "I2": "ep41~60",
        "I3": "ep61~80",
        "I4": "ep81~100",
        "I5": "ep101~120",
        "I6": "ep121~140",
        "I7": "ep141~160",
        "I8": "ep161~180",
        "I9": "ep181~200"
    }
    
    # Group the hardcoded data
    grouped_tardiness = {}
    for row in RAW_DATA:
        group = row["Group"]
        tardy = float(row["Tardiness_Cadence1"])
        
        mapped_group = group_mapping[group]
        if mapped_group not in grouped_tardiness:
            grouped_tardiness[mapped_group] = []
        grouped_tardiness[mapped_group].append(tardy)
        
    # Calculate statistics in chronological ep order
    ordered_groups = [
        "ep1~20", "ep21~40", "ep41~60", "ep61~80", "ep81~100",
        "ep101~120", "ep121~140", "ep141~160", "ep161~180", "ep181~200"
    ]
    
    summary_rows = []
    for g in ordered_groups:
        if g in grouped_tardiness:
            lst = grouped_tardiness[g]
            mean_val = get_mean(lst)
            std_val = get_std(lst)
            summary_rows.append({
                "Group": g,
                "Tardiness_Mean": round(mean_val, 2),
                "Tardiness_Std": round(std_val, 2)
            })
            
    # Overwrite the CSV file with the correct summary
    headers = ["Group", "Tardiness_Mean", "Tardiness_Std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
            
    print(f"Successfully updated and overwrote comparison CSV at: {csv_path}")
    
    # Output the result table
    print("\n================ Corrected 200ep Tardiness Results ================")
    print("| Group | Tardiness_Mean | Tardiness_Std |")
    print("| :--- | :--- | :--- |")
    for r in summary_rows:
        print(f"| {r['Group']} | {r['Tardiness_Mean']:.2f} | {r['Tardiness_Std']:.2f} |")

if __name__ == "__main__":
    main()
