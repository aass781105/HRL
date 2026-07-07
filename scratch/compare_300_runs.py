import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    global_dir = os.path.join(project_root, "plots", "global")
    print(f"Project root resolved: {project_root}")
    print(f"Global dir resolved: {global_dir}")
    print(f"Does global_dir exist? {os.path.exists(global_dir)}")
    
    if os.path.exists(global_dir):
        subdirs = [d for d in os.listdir(global_dir) if os.path.isdir(os.path.join(global_dir, d))]
        print(f"Found {len(subdirs)} subdirectories in global_dir.")
        print("First 10 subdirectories:")
        for sd in sorted(subdirs)[:10]:
            print(f"  {sd}")
            
if __name__ == "__main__":
    main()
