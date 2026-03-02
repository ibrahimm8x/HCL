"""Interactive REPL for talking to the mind."""
import sys
import argparse

from hlc.config import Config
from hlc.mind import Mind


def main():
    parser = argparse.ArgumentParser(description="Humanity's Last Creation — REPL")
    parser.add_argument("--local", action="store_true", help="Use local paths instead of Colab/GDrive")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation (architecture testing only)")
    args = parser.parse_args()

    config = Config.local() if args.local else Config()

    print("=" * 50)
    print("  Humanity's Last Creation v1")
    print("=" * 50)
    print("Initializing mind...")

    mind = Mind(config)
    stats = mind.stats()
    print(f"Mind ready. {stats['total_columns']} columns loaded.")
    print()
    print("Commands:")
    print("  seed    — Load basic knowledge (50+ facts)")
    print("  stats   — Show system statistics")
    print("  diag    — Show last routing diagnostics")
    print("  quit    — Exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "stats":
            for k, v in mind.stats().items():
                print(f"  {k}: {v}")
            continue

        if user_input.lower() == "diag":
            if mind.last_result:
                r = mind.last_result
                print(f"  Mode: {r.mode}")
                print(f"  Converged: {r.converged} ({r.iterations} iterations)")
                print(f"  Prediction error: {r.prediction_error:.4f}")
                print(f"  Active columns: {len(r.active_column_ids)}")
                print(f"  Value: {r.value_state}")
                if r.active_source_texts:
                    print(f"  Knowledge activated:")
                    for t in r.active_source_texts[:5]:
                        print(f"    - {t[:80]}...")
            else:
                print("  No queries processed yet.")
            continue

        if user_input.lower() == "seed":
            from experiments.seed_basic import BASIC_FACTS
            print(f"Seeding {len(BASIC_FACTS)} knowledge columns...")
            mind.seed_knowledge(BASIC_FACTS)
            print("Done.")
            continue

        # Process the query
        if args.no_llm:
            result = mind.process_without_llm(user_input)
            print(f"Mind (no-llm mode):")
            print(f"  Mode: {result['mode']}")
            print(f"  Converged: {result['converged']} ({result['iterations']} iters)")
            print(f"  Error: {result['prediction_error']:.4f}")
            print(f"  Value: {result['value_state']}")
            if result['active_knowledge']:
                print(f"  Activated knowledge:")
                for t in result['active_knowledge'][:5]:
                    print(f"    - {t[:80]}")
            if result['new_column_created']:
                print(f"  [New column created for this input]")
        else:
            response = mind.process(user_input)
            print(f"Mind: {response}")
            if mind.last_result:
                r = mind.last_result
                print(f"  [{r.mode} | {r.iterations} iters | "
                      f"err={r.prediction_error:.3f} | "
                      f"{r.value_state.dominant_signal.value}]")
        print()


if __name__ == "__main__":
    main()
