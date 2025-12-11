import os 

TARGET_DIR = 'assets/unified_dataset/images/val'
ZONE_IDENTIFIER = 'Zone.Identifier' 


def delete_zone_files(directory): 

    deleted_count = 0 
    print(f"Starting file cleanup in: {directory}")
    print("-" * 40)

    # os.walk generates the file names in a directory tree
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check for the specific pattern (e.g., "filename.ext:Zone.Identifier")
            if ZONE_IDENTIFIER in filename:
                file_path = os.path.join(root, filename)
                
                # --- Deletion Attempt with Error Handling ---
                try:
                    # os.remove() is used to delete the file
                    os.remove(file_path)
                    deleted_count += 1
                
                except FileNotFoundError:
                    # File may have been deleted by another process
                    print(f"SKIP: File not found (race condition): {file_path}")
                except PermissionError:
                    # File is currently locked or requires higher privileges
                    print(f"FAILED (Permission): {file_path} (Try running script as Admin)")
                except Exception as e:
                    print(f"FAILED (Unknown Error): {file_path} -> {e}")

    print("-" * 40)
    print(f"Cleanup finished. Total {deleted_count} Zone.Identifier files deleted.")

if __name__ == "__main__":
    # Safety Check: Ensure the user updates the path before running
    if 'YourUserName' in TARGET_DIR:
        print("🛑 ERROR: Please update the 'TARGET_DIRECTORY' variable with the actual path.")
    elif not os.path.exists(TARGET_DIR):
        print(f"🛑 ERROR: The specified path does not exist: {TARGET_DIR}")
    else:
        delete_zone_files(TARGET_DIR)
