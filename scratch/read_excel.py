import pandas as pd
import os

def main():
    filepath = r"C:\Users\123\Desktop\李信翰\碩一\meeting_ppt\20260708\新增 Microsoft Excel 工作表.xlsx"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    xl = pd.ExcelFile(filepath)
    df = xl.parse(xl.sheet_names[0])
    # Print the entire dataframe
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
