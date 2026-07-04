import os
import pandas as pd

def main():
    excel_path = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\換題差異\tardiness_comparsion.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"File not found: {excel_path}")
        return
        
    print(f"Loading Excel file: {excel_path}")
    excel_file = pd.ExcelFile(excel_path)
    print(f"Sheet names: {excel_file.sheet_names}\n")
    
    for sheet_name in excel_file.sheet_names:
        print(f"--- Sheet: {sheet_name} ---")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns:", list(df.columns))
        print("\nFirst 10 rows:")
        print(df.head(10).to_string())
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
