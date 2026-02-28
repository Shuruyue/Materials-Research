"""
Test Pymatviz Integration
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Add pymatviz to path (reference)
pymatviz_path = Path(__file__).resolve().parent.parent.parent / "references" / "lan496" / "pymatviz"
if pymatviz_path.exists():
    sys.path.insert(0, str(pymatviz_path))
else:
    print(f"Warning: pymatviz path not found at {pymatviz_path}")

try:
    import pymatviz
    print(f"pymatviz imported successfully: {pymatviz.__file__}")
except ImportError as e:
    print(f"Failed to import pymatviz: {e}")
    sys.exit(1)

import pandas as pd
from pymatviz import ptable_heatmap


def test_ptable_heatmap():
    print("\n--- Testing Periodic Table Heatmap ---")
    data = {"Fe": 10, "O": 20, "Ti": 5, "Sr": 5}
    series = pd.Series(data)

    try:
        fig = ptable_heatmap(series)
        output_file = Path("ptable_heatmap.png")
        fig.write_image(output_file) if hasattr(fig, 'write_image') else fig.savefig(output_file)
        print(f"Saved heatmap to {output_file}")
    except Exception as e:
        print(f"Heatmap failed: {e}")

def main():
    test_ptable_heatmap()

if __name__ == "__main__":
    main()
