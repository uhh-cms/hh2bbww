#!/usr/bin/env python3
"""
Load a correctionlib correction file and open an IPython debugging session.
Usage: python debug_corrections.py <path_to_correction_file.json.gz>
"""

import sys
import correctionlib
import IPython


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_corrections.py <path_to_correction_file.json.gz>")
        sys.exit(1)

    correction_file = sys.argv[1]
    print(f"Loading corrections from: {correction_file}")

    try:
        correction_set = correctionlib.CorrectionSet.from_file(correction_file)
        print(f"Successfully loaded correction set with {len(correction_set)} corrections")
        print("\nAvailable corrections:")
        for key in sorted(correction_set.keys()):
            print(f"  - {key}")
        if correction_set.compound:
            print("\nCompound corrections:")
            for key in sorted(correction_set.compound.keys()):
                print(f"  - {key}")
    except Exception as e:
        print(f"Error loading correction file: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Starting IPython debugging session...")
    print("Available variables: correction_set")
    print("=" * 60 + "\n")

    IPython.embed()


if __name__ == "__main__":
    main()
