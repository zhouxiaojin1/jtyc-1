
import sys
print("Subprocess Python:", sys.executable)

errors = []
packages = ['torch', 'lightgbm', 'tbats', 'pandas', 'numpy']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  Import {pkg}: OK")
    except ImportError as e:
        print(f"  Import {pkg}: FAILED - {e}")
        errors.append(pkg)

if errors:
    print("FAILED packages:", errors)
    sys.exit(1)
else:
    print("All imports successful!")
    sys.exit(0)
