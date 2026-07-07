import os

def main():
    desktop = r"C:\Users\123\Desktop"
    found = []
    print("Searching for folders containing '20260706_1' on Desktop...")
    for root, dirs, files in os.walk(desktop):
        # Skip large system or build directories to make it fast
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'trained_network', 'portable_ortools_run')]
        for d in dirs:
            if d.startswith("20260706_1"):
                found.append(os.path.join(root, d))
                
    print(f"\nFound {len(found)} folders:")
    for path in sorted(found):
        print(path)

if __name__ == "__main__":
    main()
