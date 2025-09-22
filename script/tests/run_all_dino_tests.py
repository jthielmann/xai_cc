import importlib
import sys
from pathlib import Path
# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import traceback

TESTS = [
    ("heads_and_loss", "script.tests.test_dino_heads_and_loss"),
    ("temp_and_gram", "script.tests.test_dino_temp_and_gram"),
    ("optim_schedules", "script.tests.test_dino_optim_and_schedules"),
    ("features_cache", "script.tests.test_dino_features_and_cache"),
    ("stain_norm", "script.tests.test_stain_norm"),
]


def main():
    passed, failed = 0, 0
    for name, modname in TESTS:
        print(f"\n=== [{name}] START ===")
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, 'run'):
                mod.run()
            else:
                print(f"[WARN] {modname} has no run(); importing only")
            print(f"=== [{name}] PASS ===")
            passed += 1
        except SystemExit as e:
            print(f"=== [{name}] EXIT {e.code} ===")
        except Exception:
            print(f"=== [{name}] FAIL ===")
            traceback.print_exc()
            failed += 1
    print(f"\nRESULT: passed={passed} failed={failed}")


if __name__ == '__main__':
    main()
