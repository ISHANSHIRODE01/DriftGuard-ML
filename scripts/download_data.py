"""Fetch ELEC2 from the river package and persist as CSV."""
from pathlib import Path
import pandas as pd
from river import datasets

def main(out: str = "data/elec2.csv") -> None:
    rows = []
    for x, y in datasets.Elec2():
        d = dict(x); d["target"] = int(y); rows.append(d)
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"ELEC2 -> {out}  shape={df.shape}  positive_rate={df.target.mean():.4f}")

if __name__ == "__main__":
    main()
