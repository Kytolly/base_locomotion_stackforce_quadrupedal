#!/usr/bin/env python3
"""
Monitor training health metrics for Stackforce Quadrupedal Locomotion.
Standards defined in docs/TRAINING_HEALTH_METRICS.md.
"""

import os
import re
import sys
import glob
import json
import subprocess
from datetime import datetime

LOG_FILE = "wandb/latest-run/files/output.log"
CONFIG_PATH = "configs/train/base_locomotion_hybrid.yaml"

def get_gpu_status():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        line = res.stdout.strip().split("\n")[0]
        name, mem_used, mem_total, temp, util, pwr = [x.strip() for x in line.split(",")]
        return {
            "name": name,
            "mem_used_mib": float(mem_used),
            "mem_total_mib": float(mem_total),
            "mem_percent": float(mem_used) / float(mem_total) * 100,
            "temp_c": float(temp),
            "util_percent": float(util),
            "power_w": float(pwr)
        }
    except Exception as e:
        return {"error": str(e)}

def parse_iterations(log_path):
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    parts = re.split(r"Learning iteration\s+(\d+)/(\d+)", content)
    if len(parts) < 4:
        return []

    iterations = []
    for i in range(1, len(parts), 3):
        iter_num = int(parts[i])
        total_iters = int(parts[i+1])
        body = parts[i+2]

        metrics = {
            "iteration": iter_num,
            "total_iterations": total_iters,
        }

        for line in body.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not v:
                continue

            try:
                if v.endswith("s"):
                    metrics[k] = float(v[:-1])
                elif re.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$", v):
                    metrics[k] = float(v)
                elif ":" in v: # time format
                    metrics[k] = v
                else:
                    metrics[k] = v
            except ValueError:
                metrics[k] = v

        iterations.append(metrics)
    return iterations

def analyze_health(iterations, gpu_info):
    if not iterations:
        return {"status": "NO_DATA", "message": "No iteration data found."}

    latest = iterations[-1]
    total_iters = latest["total_iterations"]
    current_iter = latest["iteration"]
    progress_ratio = current_iter / total_iters if total_iters > 0 else 0

    # Sliding window (100 iter or available)
    w_size = min(100, len(iterations))
    window = iterations[-w_size:]

    def window_mean(key):
        vals = [it[key] for it in window if key in it and isinstance(it[key], (int, float))]
        return sum(vals) / len(vals) if vals else None

    # Red alerts checks
    red_alerts = []
    warnings = []

    # 1. Runtime Health
    all_finite = latest.get("train/runtime/all_finite", None)
    finite_rate = latest.get("train/runtime/finite_rate", None)
    if all_finite is not None and all_finite < 1.0:
        red_alerts.append(f"CRITICAL: train/runtime/all_finite = {all_finite} < 1.0 (Non-finite numbers encountered!)")
    if finite_rate is not None and finite_rate < 1.0:
        red_alerts.append(f"CRITICAL: train/runtime/finite_rate = {finite_rate*100:.1f}% < 100%")

    for loss_key in ["Mean value loss", "Mean surrogate loss", "Mean reward"]:
        val = latest.get(loss_key, None)
        if val is not None and (val != val or abs(val) > 1e6):
            red_alerts.append(f"CRITICAL: {loss_key} diverged: {val}")

    if "mem_percent" in gpu_info and gpu_info["mem_percent"] > 90.0:
        red_alerts.append(f"CRITICAL: GPU Memory {gpu_info['mem_percent']:.1f}% > 90%")

    # 2. Actuation Health
    act_sat_latest = latest.get("train/actuation/action_saturation_rate", 0.0)
    act_sat_win = window_mean("train/actuation/action_saturation_rate") or act_sat_latest

    if act_sat_win > 0.50:
        red_alerts.append(f"CRITICAL: Action saturation 100-iter avg {act_sat_win*100:.1f}% > 50%")
    elif act_sat_win > 0.20:
        warnings.append(f"WARNING: Action saturation 100-iter avg {act_sat_win*100:.1f}% > 20% (unhealthy)")
    elif act_sat_win >= 0.10:
        warnings.append(f"NOTICE: Action saturation 100-iter avg {act_sat_win*100:.1f}% in [10%, 20%] (observing)")

    # 3. Stage-specific criteria
    max_ep_len = 1500.0
    ep_len = window_mean("Mean episode length") or latest.get("Mean episode length", 0.0)
    unsafe_term = window_mean("train/safety/unsafe_termination") or latest.get("train/safety/unsafe_termination", 0.0)
    timeout = window_mean("train/safety/timeout") or latest.get("train/safety/timeout", 0.0)
    fwd_rmse = window_mean("train/locomotion/forward_velocity_rmse_mps") or latest.get("train/locomotion/forward_velocity_rmse_mps", 999.0)
    lat_rmse = window_mean("train/locomotion/lateral_velocity_rmse_mps") or latest.get("train/locomotion/lateral_velocity_rmse_mps", 999.0)
    yaw_rmse = window_mean("train/locomotion/yaw_rate_rmse_radps") or latest.get("train/locomotion/yaw_rate_rmse_radps", 999.0)
    height_rmse = window_mean("train/locomotion/body_height_rmse_m") or latest.get("train/locomotion/body_height_rmse_m", 999.0)
    wheel_contact = window_mean("train/support/wheel_contact_fraction") or latest.get("train/support/wheel_contact_fraction", 0.0)
    tilt_p95 = window_mean("train/safety/base_tilt_max_p95_rad") or latest.get("train/safety/base_tilt_max_p95_rad", 0.0)
    coll_rate = window_mean("train/safety/base_collision_rate") or latest.get("train/safety/base_collision_rate", 0.0)
    inv_rate = window_mean("train/support/invalid_rate") or latest.get("train/support/invalid_rate", 0.0)

    stage = "Early (0-20%)" if progress_ratio < 0.20 else ("Mid (20-70%)" if progress_ratio < 0.70 else "Late (70-100%)")

    stage_checks = {}
    if stage == "Early (0-20%)":
        stage_checks["Episode Length > 50%"] = (ep_len > max_ep_len * 0.5, f"{ep_len:.1f} / {max_ep_len*0.5:.0f}")
        stage_checks["Unsafe Termination < 30%"] = (unsafe_term < 0.30, f"{unsafe_term*100:.1f}% (target < 30%)")
        stage_checks["Forward RMSE < 0.30 m/s"] = (fwd_rmse < 0.30, f"{fwd_rmse:.3f} m/s (target < 0.30)")
    elif stage == "Mid (20-70%)":
        stage_checks["Episode Length > 90%"] = (ep_len > max_ep_len * 0.9, f"{ep_len:.1f} / {max_ep_len*0.9:.0f}")
        stage_checks["Timeout > 90%"] = (timeout > 0.90, f"{timeout*100:.1f}% (target > 90%)")
        stage_checks["Unsafe Termination < 10%"] = (unsafe_term < 0.10, f"{unsafe_term*100:.1f}% (target < 10%)")
        stage_checks["Forward RMSE < 0.22 m/s"] = (fwd_rmse < 0.22, f"{fwd_rmse:.3f} m/s (target < 0.22)")
        stage_checks["Lateral RMSE < 0.12 m/s"] = (lat_rmse < 0.12, f"{lat_rmse:.3f} m/s (target < 0.12)")
        stage_checks["Yaw Rate RMSE < 0.25 rad/s"] = (yaw_rmse < 0.25, f"{yaw_rmse:.3f} rad/s (target < 0.25)")
    else:
        stage_checks["Timeout > 95%"] = (timeout > 0.95, f"{timeout*100:.1f}% (target > 95%)")
        stage_checks["Unsafe Termination <= 5%"] = (unsafe_term <= 0.05, f"{unsafe_term*100:.1f}% (target <= 5%)")
        stage_checks["Forward RMSE <= 0.15 m/s"] = (fwd_rmse <= 0.15, f"{fwd_rmse:.3f} m/s (target <= 0.15)")
        stage_checks["Lateral RMSE <= 0.12 m/s"] = (lat_rmse <= 0.12, f"{lat_rmse:.3f} m/s (target <= 0.12)")
        stage_checks["Yaw Rate RMSE <= 0.10 rad/s"] = (yaw_rmse <= 0.10, f"{yaw_rmse:.3f} rad/s (target <= 0.10)")
        stage_checks["Body Height RMSE <= 0.025 m"] = (height_rmse <= 0.025, f"{height_rmse:.3f} m (target <= 0.025)")

    stability_checks = {
        "Wheel Contact Fraction >= 50%": (wheel_contact >= 0.50, f"{wheel_contact*100:.1f}% (target >= 50%)"),
        "Invalid Local Support <= 1%": (inv_rate <= 0.01, f"{inv_rate*100:.2f}% (target <= 1%)"),
        "Base Collision Rate <= 1%": (coll_rate <= 0.01, f"{coll_rate*100:.2f}% (target <= 1%)"),
        "Base Tilt Max p95 <= 1.05 rad": (tilt_p95 <= 1.05, f"{tilt_p95:.3f} rad (target <= 1.05)"),
        "Body Height RMSE <= 0.025 m": (height_rmse <= 0.025, f"{height_rmse:.3f} m (target <= 0.025)")
    }

    # Window summary dict
    window_summary = {
        "window_size": w_size,
        "action_saturation_avg": act_sat_win,
        "episode_length_avg": ep_len,
        "forward_rmse_avg": fwd_rmse,
        "unsafe_termination_avg": unsafe_term,
        "timeout_avg": timeout,
        "wheel_contact_avg": wheel_contact,
        "collision_rate_avg": coll_rate
    }

    # Overall trends (first 10 vs last 10)
    trends = {}
    if len(iterations) >= 10:
        first_10 = iterations[:10]
        last_10 = iterations[-10:]
        def mean_k(sub, k):
            v = [x[k] for x in sub if k in x and isinstance(x[k], (int, float))]
            return sum(v)/len(v) if v else None

        for k in ["Mean reward", "Mean episode length", "train/locomotion/forward_velocity_rmse_mps",
                  "train/safety/unsafe_termination", "train/safety/timeout", "train/terrain/level"]:
            m_init = mean_k(first_10, k)
            m_curr = mean_k(last_10, k)
            if m_init is not None and m_curr is not None:
                diff = m_curr - m_init
                trends[k] = {"initial": m_init, "current": m_curr, "diff": diff}

    return {
        "current_iter": current_iter,
        "total_iters": total_iters,
        "progress_percent": progress_ratio * 100,
        "stage": stage,
        "latest": latest,
        "window_summary": window_summary,
        "gpu": gpu_info,
        "red_alerts": red_alerts,
        "warnings": warnings,
        "stage_checks": stage_checks,
        "stability_checks": stability_checks,
        "trends": trends,
        "num_iters_collected": len(iterations)
    }

def print_report(res):
    print("=" * 80)
    print(f"训练健康度监控报告 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("=" * 80)

    cur = res["current_iter"]
    tot = res["total_iters"]
    pct = res["progress_percent"]
    stage = res["stage"]
    w_sum = res["window_summary"]
    w_size = w_sum["window_size"]
    print(f"进度: Iteration {cur}/{tot} ({pct:.2f}%) | 当前阶段: {stage}")

    latest = res["latest"]
    eta = latest.get("ETA", "N/A")
    iter_time = latest.get("Iteration time", "N/A")
    print(f"单步耗时: {iter_time} | 预计剩余时间 (ETA): {eta}")

    gpu = res["gpu"]
    if "error" not in gpu:
        print(f"GPU: {gpu['name']} | 显存: {gpu['mem_used_mib']:.0f}/{gpu['mem_total_mib']:.0f} MiB ({gpu['mem_percent']:.1f}%) | 温度: {gpu['temp_c']}°C | 功率: {gpu['power_w']} W")
    print("-" * 80)

    print("【1. 运行时健康 (Runtime Health)】")
    all_finite = latest.get("train/runtime/all_finite", "N/A")
    finite_rate = latest.get("train/runtime/finite_rate", "N/A")
    print(f"  • all_finite:   {all_finite}  (标准: 始终 1.0) -> {'🟢 正常' if all_finite == 1.0 else '🔴 异常'}")
    print(f"  • finite_rate:  {finite_rate if isinstance(finite_rate, str) else f'{finite_rate*100:.1f}%'} (标准: 100%) -> {'🟢 正常' if finite_rate == 1.0 else '🔴 异常'}")
    print(f"  • Mean value loss:     {latest.get('Mean value loss', 'N/A')}")
    print(f"  • Mean surrogate loss: {latest.get('Mean surrogate loss', 'N/A')}")
    print(f"  • Mean reward:         {latest.get('Mean reward', 'N/A')}")
    print(f"  • Mean episode length: {latest.get('Mean episode length', 'N/A')} steps (滑动平均: {w_sum['episode_length_avg']:.1f})")

    print(f"\n【2. 阶段参考线与学习趋势 (基于滑动窗口 {w_size} iters)】")
    for check_name, (passed, info) in res["stage_checks"].items():
        tag = "🟢 达标" if passed else "🟡 进展中 / 观察"
        print(f"  • {check_name:<28}: {info} -> {tag}")

    print("\n【3. 动作与执行器健康 (Actuation Health)】")
    sat_curr = latest.get("train/actuation/action_saturation_rate", 0.0)
    sat_win = w_sum["action_saturation_avg"]
    leg_rms = latest.get("train/actuation/leg_action_rms", "N/A")
    wheel_rms = latest.get("train/actuation/wheel_action_rms", "N/A")
    act_rate_rms = latest.get("train/actuation/action_rate_rms", "N/A")
    sat_tag = "🟢 健康 (<10%)" if sat_win < 0.10 else ("🟡 观察 (10-20%)" if sat_win <= 0.20 else "🔴 不健康 (>20%)")
    print(f"  • Action Saturation (当前/滑动): {sat_curr*100:.2f}% / {sat_win*100:.2f}% -> {sat_tag}")
    print(f"  • Leg Action RMS:         {leg_rms}")
    print(f"  • Wheel Action RMS:       {wheel_rms}")
    print(f"  • Action Rate RMS:        {act_rate_rms}")

    print(f"\n【4. 支撑、接触与稳定性 (基于滑动窗口 {w_size} iters)】")
    for check_name, (passed, info) in res["stability_checks"].items():
        tag = "🟢 达标" if passed else "🟡 待改善"
        print(f"  • {check_name:<30}: {info} -> {tag}")

    print("\n【5. 地形课程进展 (Terrain Curriculum)】")
    print(f"  • Terrain Level:        {latest.get('train/terrain/level', 'N/A')}")
    print(f"  • Difficulty Midpoint:  {latest.get('train/terrain/difficulty_midpoint', 'N/A')}")

    if res["trends"]:
        print("\n【6. 宏观训练动态趋势 (前10 vs 最近10 iters)】")
        for k, t in res["trends"].items():
            short_name = k.replace("train/", "")
            diff = t["diff"]
            sign = "+" if diff >= 0 else ""
            print(f"  • {short_name:<35}: {t['initial']:.3f} -> {t['current']:.3f} ({sign}{diff:.3f})")

    print("-" * 80)
    if res["red_alerts"]:
        print("🚨 红色警报 (RED ALERTS):")
        for a in res["red_alerts"]:
            print(f"  - {a}")
    else:
        print("✅ 无红色警报 (No Red Alerts)，运行时数值稳定。")

    if res["warnings"]:
        print("⚠️ 注意事项 (Warnings / Observations):")
        for w in res["warnings"]:
            print(f"  - {w}")
    print("=" * 80)

if __name__ == "__main__":
    gpu = get_gpu_status()
    iters = parse_iterations(LOG_FILE)
    res = analyze_health(iters, gpu)
    print_report(res)
    if "--json" in sys.argv:
        print("\nJSON_OUTPUT:")
        print(json.dumps(res, indent=2, default=str))
