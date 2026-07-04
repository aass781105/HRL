import os
import pandas as pd

def main():
    excel_path = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\換題差異\tardiness_comparsion.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"File not found: {excel_path}")
        return
        
    print(f"Reading Excel file: {excel_path}")
    # Read sheet "工作表1"
    df = pd.read_excel(excel_path, sheet_name="工作表1")
    
    # Map seed to values
    data_8env = {
        1: "2329 / 15",
        2: "9444 / 12",
        3: "871 / 17",
        4: "2624 / 14",
        5: "10772 / 14",
        6: "2906 / 16",
        7: "3844 / 13",
        8: "13570 / 13",
        9: "571 / 18",
        10: "3569 / 14"
    }
    
    data_Rstab = {
        1: "1640 / 15",
        2: "10276 / 14",
        3: "385 / 20",
        4: "857 / 18",
        5: "7180 / 16",
        6: "2638 / 20",
        7: "2045 / 18",
        8: "11141 / 14",
        9: "623 / 23",
        10: "860 / 16"
    }
    
    df["8env"] = df["seed"].map(data_8env)
    df["Rstab"] = df["seed"].map(data_Rstab)
    
    # Save back to the Excel file
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="工作表1", index=False)
        
    print("Successfully updated Excel file with '8env' and 'Rstab' columns.")
    print("\nUpdated Excel contents:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
