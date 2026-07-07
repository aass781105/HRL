import os

def main():
    project_root = os.getcwd()
    global_dir = os.path.join(project_root, "plots", "global")
    
    if os.path.exists(global_dir):
        all_dirs = sorted(os.listdir(global_dir))
        print(f"Total subdirectories: {len(all_dirs)}")
        for d in all_dirs:
            print(d)
            
if __name__ == "__main__":
    main()
