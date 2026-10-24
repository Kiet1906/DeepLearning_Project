
import sys
import platform

def main():
    print("=" * 45)
    print("DRIVER DROWSINESS DETECTION")
    print("STEP 1 - ENVIRONMENT CHECK")
    print("=" * 45)

    print(f"Python version : {sys.version.split()[0]}")
    print(f"Python path    : {sys.executable}")
    print(f"Operating sys  : {platform.system()}")

    print("\nEnvironment check completed.")

if __name__ == "__main__":
    main()
