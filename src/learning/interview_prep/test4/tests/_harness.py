"""Tiny test runner so the graders need nothing but torch. Do not edit."""

import os
import sys
import traceback

_SRC = os.environ.get(
    "ASSESSMENT_SRC",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
sys.path.insert(0, _SRC)


def run(tests):
    passed, todo = 0, 0
    for name, fn in tests:
        try:
            fn()
        except NotImplementedError:
            print(f"  TODO {name} (not implemented yet)")
            todo += 1
        except Exception as e:
            line = traceback.format_exc().strip().splitlines()[-1]
            print(f"  FAIL {name}: {line}")
        else:
            print(f"  PASS {name}")
            passed += 1
    print(f"\n{passed}/{len(tests)} passed"
          + (f" ({todo} not implemented)" if todo else ""))
    return passed == len(tests)
