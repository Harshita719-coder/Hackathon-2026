import pandas as pd


def build_search_space(csv_path: str) -> dict:
    """
    Build search space from the real CSV columns.
    """

    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].copy()

    search_space = {
        "ligand": sorted(train_df["ligand"].dropna().unique().tolist()),
        "additive": sorted(train_df["additive"].dropna().unique().tolist()),
        "base": sorted(train_df["base"].dropna().unique().tolist()),
        "aryl_halide": sorted(train_df["aryl_halide"].dropna().unique().tolist()),
    }

    return search_space


def recommend_next_conditions(csv_path: str, aryl_halide: str, top_k: int = 5):
    """
    Recommend strong candidate reactions for the same aryl_halide.

    In later phases, this can be replaced with BayBE.
    For Phase 1, this gives a valid suggestion round using the real dataset.
    """

    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"].copy()

    subset = train_df[train_df["aryl_halide"] == aryl_halide].copy()

    if subset.empty:
        subset = train_df.copy()

    subset = subset.sort_values(by="yield", ascending=False)

    recommendations = []

    for _, row in subset.head(top_k).iterrows():
        recommendations.append(
            {
                "ligand": row["ligand"],
                "additive": row["additive"],
                "base": row["base"],
                "aryl_halide": row["aryl_halide"],
                "expected_yield_percent": round(float(row["yield"]), 2),
            }
        )

    return recommendations


if __name__ == "__main__":
    csv_path = r"C:\Users\vrush\Downloads\buchwald_hartwig_huggingface.csv"

    print("Building search space...\n")

    space = build_search_space(csv_path)

    for key, values in space.items():
        print(f"{key}: {len(values)} options")

    print("\nSearch space built successfully.")