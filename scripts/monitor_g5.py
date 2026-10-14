import json
import time
import subprocess
import os
import re
from pathlib import Path

ROOT = Path("/home/kytolly/Project/IsaacProject/base_locomotion_stackforce_quadrupedal")
G5_DIR = ROOT / "output/fix_plan/G5"
MATRIX_FILE = G5_DIR / "timing_matrix.json"

ITEMS_ORDER = ["baseline", "T2", "T3a", "T3b", "T4", "T5", "T6", "T7"]

def get_running_train():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,etimes,args"], text=True)
        train_pids = []
        for line in out.splitlines():
            if "scripts/rsl_rl/train.py" in line:
                parts = line.strip().split(None, 2)
                pid = int(parts[0])
                etimes = int(parts[1])
                args = parts[2]
                match = re.search(r"agent\.run_name=[^\s]+-g5-([a-zA-Z0-9]+)-seed", args)
                item = match.group(1).upper() if match else "unknown"
                if item == "BASELINE":
                    item = "baseline"
                train_pids.append((pid, etimes, item, args))
        return train_pids
    except Exception as e:
        return []

def get_realtime_metrics(item):
    log_file = G5_DIR / f"{item}.log"
    if not log_file.exists():
        return None
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        matches = re.findall(r'Learning iteration (\d+)/(\d+).*?Collection time:\s*([\d\.]+)s.*?Steps per second:\s*(\d+)', text, re.DOTALL)
        eta_matches = re.findall(r'ETA:\s*([0-9:]+)', text)
        eta = eta_matches[-1] if eta_matches else None

        if matches:
            last_it, total_it, col_time, sps = matches[-1]
            return {
                "iteration": f"{last_it}/{total_it}",
                "col_time": float(col_time),
                "sps": int(sps),
                "eta": eta
            }
    except Exception:
        pass
    return None

def load_matrix():
    if MATRIX_FILE.exists():
        try:
            return json.loads(MATRIX_FILE.read_text())
        except Exception:
            return {}
    return {}

def status_report():
    matrix = load_matrix()
    items_data = matrix.get("items", {})
    running = get_running_train()

    print("=" * 88)
    print(f"Isaac G5 Throughput Matrix Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)

    running_items = {r[2].lower(): r for r in running}
    base_time = items_data.get("baseline", {}).get("collection_time_s")

    print(f"{'Item':<9} | {'Status':<10} | {'Col Time (s)':<13} | {'Gain vs Base':<13} | {'SPS':<7} | {'Details / Progress'}")
    print("-" * 88)

    for item in ITEMS_ORDER:
        data = items_data.get(item, {})
        col_time = data.get("collection_time_s")
        gain = data.get("gain_vs_baseline")

        status = "Completed" if col_time is not None else "Pending"
        sps_str = "-"
        details = ""

        if item.lower() in running_items:
            status = "RUNNING"
            pid, etimes, _, _ = running_items[item.lower()]
            metrics = get_realtime_metrics(item)
            if metrics:
                col_time_str = f"~{metrics['col_time']:.3f}s"
                if base_time:
                    cur_gain = (base_time - metrics['col_time']) / base_time
                    gain_str = f"~{cur_gain*100:+.2f}%"
                else:
                    gain_str = "-"
                sps_str = str(metrics['sps'])
                details = f"Iter {metrics['iteration']}, ETA: {metrics['eta']}, PID {pid}"
            else:
                col_time_str = "-"
                gain_str = "-"
                details = f"PID {pid} (init/syncing)"
        elif status == "Completed":
            gain_str = f"{gain*100:+.2f}%" if gain is not None else ("(Base)" if item == "baseline" else "-")
            col_time_str = f"{col_time:.3f}s" if col_time is not None else "-"
            details = data.get("description", "")
        else:
            gain_str = "-"
            col_time_str = "-"
            details = ""

        print(f"{item:<9} | {status:<10} | {col_time_str:<13} | {gain_str:<13} | {sps_str:<7} | {details}")

    print("-" * 88)
    if running:
        for pid, etimes, item, args in running:
            print(f"Active Isaac Process: PID {pid} | Target: {item} | Elapsed: {etimes//60}m {etimes%60}s")
    else:
        print("No active rsl_rl/train.py process running.")
    print("=" * 88)

if __name__ == "__main__":
    status_report()
