from pathlib import Path
import importlib.util
import pandas as pd


CSV_PATH = r"C:\Users\vrush\Downloads\buchwald_hartwig_huggingface.csv"

REQUIRED_PACKAGES = ["streamlit", "pandas", "numpy"]
REQUIRED_COLUMNS = ["ligand", "additive", "base", "aryl_halide", "yield", "split"]


def check_package(package_name):
    return importlib.util.find_spec(package_name) is not None


def main():
    print("=" * 60)
    print("PHASE 5 PREFLIGHT CHECK")
    print("=" * 60)

    print("\n1. Package Check")
    print("-" * 60)

    all_packages_ok = True

    for package in REQUIRED_PACKAGES:
        installed = check_package(package)

        if installed:
            print(f"[OK] {package}")
        else:
            print(f"[MISSING] {package}")
            all_packages_ok = False

    print("\n2. CSV Check")
    print("-" * 60)

    path = Path(CSV_PATH)

    if not path.exists():
        print(f"[MISSING] CSV not found: {CSV_PATH}")
        return

    print(f"[OK] CSV found: {CSV_PATH}")

    df = pd.read_csv(path)

    print("\n3. Column Check")
    print("-" * 60)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return

    print("[OK] Required columns found")

    print("\n4. Dataset Stats")
    print("-" * 60)
    print(f"Rows: {len(df)}")
    print(f"Train rows: {len(df[df['split'] == 'train'])}")
    print(f"Mean yield: {round(float(df['yield'].mean()), 2)}")
    print(f"Low yield <30%: {round(float((df['yield'] < 30).mean() * 100), 2)}%")
    print(f"High yield >=80%: {round(float((df['yield'] >= 80).mean() * 100), 2)}%")

    print("\n5. Final Result")
    print("-" * 60)

    if all_packages_ok:
        print("[READY] You can run the Streamlit app.")
        print(
            r"Run: C:\Users\vrush\AppData\Local\Programs\Python\Python312\python.exe "
            r"-m streamlit run C:\Users\vrush\Downloads\app_phase5.py"
        )
    else:
        print("[NOT READY] Install missing packages first.")
        print(
            r"Run: C:\Users\vrush\AppData\Local\Programs\Python\Python312\python.exe "
            r"-m pip install streamlit pandas numpy"
        )


if __name__ == "__main__":
    main()