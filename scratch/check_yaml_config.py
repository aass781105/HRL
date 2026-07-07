import os

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_config_dir = os.path.join(project_root, "yaml_config")
    
    if os.path.exists(yaml_config_dir):
        print(f"Directory exists: {yaml_config_dir}")
        print("Files:")
        for f in os.listdir(yaml_config_dir):
            print(f"  {f}")
    else:
        print(f"Directory does not exist: {yaml_config_dir}")

if __name__ == "__main__":
    main()
