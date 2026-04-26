from Oracle import BHOracle
from baybe_init import build_search_space, recommend_next_conditions


def main():
    csv_path = r"C:\Users\vrush\Downloads\buchwald_hartwig_huggingface.csv"

    print("=" * 60)
    print("PHASE 1: DATA & ORACLE")
    print("=" * 60)

    oracle = BHOracle(csv_path)

    print("\n1. Dataset Report")
    print("-" * 60)

    report = oracle.dataset_report()

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n2. Search Space")
    print("-" * 60)

    search_space = build_search_space(csv_path)

    for parameter, values in search_space.items():
        print(f"{parameter}: {len(values)} options")

    print("\n3. Starting Failed Reaction")
    print("-" * 60)

    starting_reaction = oracle.get_low_yield_start()

    for key, value in starting_reaction.items():
        print(f"{key}: {value}")

    print("\n4. Oracle Query Result")
    print("-" * 60)

    oracle_result = oracle.query(
        ligand=starting_reaction["ligand"],
        additive=starting_reaction["additive"],
        base=starting_reaction["base"],
        aryl_halide=starting_reaction["aryl_halide"],
    )

    for key, value in oracle_result.items():
        print(f"{key}: {value}")

    print("\n5. Recommended Better Conditions")
    print("-" * 60)

    recommendations = recommend_next_conditions(
        csv_path=csv_path,
        aryl_halide=starting_reaction["aryl_halide"],
        top_k=5,
    )

    for i, rec in enumerate(recommendations, start=1):
        print(f"\nRecommendation {i}")
        print(f"ligand: {rec['ligand']}")
        print(f"additive: {rec['additive']}")
        print(f"base: {rec['base']}")
        print(f"aryl_halide: {rec['aryl_halide']}")
        print(f"expected_yield_percent: {rec['expected_yield_percent']}")

    print("\n6. Test Best Recommendation With Oracle")
    print("-" * 60)

    best = recommendations[0]

    best_result = oracle.query(
        ligand=best["ligand"],
        additive=best["additive"],
        base=best["base"],
        aryl_halide=best["aryl_halide"],
    )

    for key, value in best_result.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()