import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    global_dir = os.path.join(project_root, "plots", "global")
    
    if os.path.exists(global_dir):
        subdirs = [d for d in os.listdir(global_dir) if os.path.isdir(os.path.join(global_dir, d))]
        matching = [d for d in subdirs if d.startswith("20260706")]
        print(f"Found {len(matching)} folders matching date 20260706:")
        for m in sorted(matching):
            print(f"  {m}")
            
if __name__ == "__main__":
    main()
