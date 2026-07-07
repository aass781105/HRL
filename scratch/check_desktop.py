import os

def main():
    desktop = r"C:\Users\123\Desktop"
    if os.path.exists(desktop):
        print("Desktop directories:")
        for d in os.listdir(desktop):
            print(f"  {d}")
            
    print("\nCurrent working directory (getcwd):", os.getcwd())
    
if __name__ == "__main__":
    main()
