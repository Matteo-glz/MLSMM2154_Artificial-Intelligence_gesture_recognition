from datetime import datetime
import os
import numpy as np 


def save_results(summary, best_config, cm, df, config_label, output_dir="results"):
    """Write summary + confusion matrix to a txt file, and raw results to csv."""
    os.makedirs(output_dir, exist_ok=True)
    safe_label = config_label.replace(" ", "_")

    # ── txt report ───────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, f"{safe_label}.txt")
    with open(txt_path, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"RESULTS — {config_label}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write("FULL SUMMARY (mean accuracy ± std per config)\n")
        f.write("-"*40 + "\n")
        f.write(summary.to_string())
        f.write("\n\n")

        f.write(f"BEST CONFIG: {best_config}\n")
        best_mean = summary.loc[best_config, "mean"]
        best_std  = summary.loc[best_config, "std"]
        f.write(f"Mean accuracy : {best_mean:.4f}\n")
        f.write(f"Std           : {best_std:.4f}\n\n")

        f.write("CONFUSION MATRIX (best config)\n")
        f.write("-"*40 + "\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n")

    # ── csv raw results ───────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"{safe_label}_raw.csv")
    df.to_csv(csv_path, index=False)

    print(f"  -> Saved: {txt_path}")
    print(f"  -> Saved: {csv_path}")