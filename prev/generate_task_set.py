from task import SegInfTask
import json
import os
import pickle
import random
import ast
import math
import shutil

from utils import load_yaml

# Reference: Analytical Enhancements and Practical Insights for MPCP with Self-Suspensions

CONFIG_PATH = "generate_task_set.yaml"
_log_lines = []

def _log(line):
    _log_lines.append(line)


def UUniFast(n, U_total):
    """
    UUniFast algorithm to generate n utilization values that sum to U_total.
    """
    utilizations = []
    sum_U = U_total
    for i in range(1, n):
        next_sum_U = sum_U * (random.random() ** (1 / (n - i)))
        utilizations.append(sum_U - next_sum_U)
        sum_U = next_sum_U
    utilizations.append(sum_U)
    return utilizations


def split_int(C, n):
    if n <= 0:
        return []

    if C <= 0:
        return [0] * n

    if n > C:
        parts = [0] * n
        for idx in random.sample(range(n), C):
            parts[idx] = 1
        return parts

    cuts = sorted(random.sample(range(1, C), n - 1))
    parts = []
    prev = 0

    for c in cuts:
        parts.append(c - prev)
        prev = c

    parts.append(C - prev)
    return parts



def _as_pair(value, name, cast):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a list/tuple of length 2")
    return (cast(value[0]), cast(value[1]))


config = load_yaml(CONFIG_PATH)

OUTPUT_DIR = config.get("output_dir", "task_set_list/utilization")
N = int(config.get("n_task_sets", 500))

number_of_cpu_range = _as_pair(config.get("number_of_cpu_range", [1, 3]), "number_of_cpu_range", int)
if "utilization_range" in config:
    utilization_range = _as_pair(config.get("utilization_range"), "utilization_range", float)
    utilization_is_total = True
else:
    # Backward compatibility for older configs.
    utilization_range = _as_pair(config.get("utilization_per_cpu_range", [0.1, 1.0]), "utilization_per_cpu_range", float)
    utilization_is_total = False
utilization_step = float(config.get("utilization_step", 0.1))
number_of_tasks_per_cpu_range = _as_pair(config.get("number_of_tasks_per_cpu_range", [3, 3]), "number_of_tasks_per_cpu_range", int)
period_range = _as_pair(config.get("period_range", [100, 10000]), "period_range", int)
G_ratio_range = _as_pair(config.get("G_ratio_range", [0.1, 0.8]), "G_ratio_range", float)
if G_ratio_range[0] > G_ratio_range[1]:
    raise ValueError("G_ratio_range[0] must be <= G_ratio_range[1]")
if G_ratio_range[0] < 0.0 or G_ratio_range[1] > 1.0:
    raise ValueError("G_ratio_range values must be in [0.0, 1.0]")
number_of_inference_segments_range = _as_pair(config.get("number_of_inference_segments_range", [1, 1]), "number_of_inference_segments_range", int)
max_block_count_range = _as_pair(config.get("max_block_count_range", [10, 100]), "max_block_count_range", int)
G_utilization_threshold_range = _as_pair(config.get("G_utilization_threshold_range", [1.0, 1.0]), "G_utilization_threshold_range", float)
per_splitting_overhead = int(config.get("per_splitting_overhead", 10))
uniform_task_utilization = bool(config.get("uniform_task_utilization", False))
uniform_cpu_utilization = bool(config.get("uniform_cpu_utilization", True))

"""
Generate task sets with varying utilization
"""
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start from a clean output directory to avoid mixing old utilization files.
for entry in os.listdir(OUTPUT_DIR):
    entry_path = os.path.join(OUTPUT_DIR, entry)
    if os.path.isfile(entry_path) or os.path.islink(entry_path):
        os.unlink(entry_path)
    elif os.path.isdir(entry_path):
        shutil.rmtree(entry_path)

# Save generation config snapshot with outputs for reproducibility.
if os.path.exists(CONFIG_PATH):
    shutil.copy2(CONFIG_PATH, os.path.join(OUTPUT_DIR, os.path.basename(CONFIG_PATH)))

if utilization_step <= 0:
    raise ValueError("utilization_step must be > 0")

u_start = float(utilization_range[0])
u_end = float(utilization_range[1])
if u_start > u_end:
    raise ValueError("utilization_range[0] must be <= utilization_range[1]")

# Align utilization points to multiples of `utilization_step` from 0.0.
# Example: range [0.1, 1.0], step 0.2 -> 0.2, 0.4, 0.6, 0.8, 1.0
first_k = int(math.ceil((u_start - 1e-12) / utilization_step))
last_k = int(math.floor((u_end + 1e-12) / utilization_step))
utilization_values = [round(k * utilization_step, 10) for k in range(first_k, last_k + 1)]

for U_selected in utilization_values:
    task_set_list_data = {
        "parameter_ranges": {
            "number_of_cpu_range": number_of_cpu_range,
            "utilization_range": utilization_range,
            "number_of_tasks_per_cpu_range": number_of_tasks_per_cpu_range,
            "period_range": period_range,
            "G_ratio_range": G_ratio_range,
            "number_of_inference_segments_range": number_of_inference_segments_range,
            "max_block_count_range": max_block_count_range,
            "G_utilization_threshold_range": G_utilization_threshold_range,
            "per_splitting_overhead": per_splitting_overhead,
            "uniform_task_utilization": uniform_task_utilization,
            "uniform_cpu_utilization": uniform_cpu_utilization,
        },
        "selected_utilization": U_selected,
        "selected_utilization_kind": "total" if utilization_is_total else "per_cpu",
        "task_set_list": []
    }
    task_set_list_pickle_data = {
        "parameter_ranges": task_set_list_data["parameter_ranges"],
        "selected_utilization": U_selected,
        "selected_utilization_kind": task_set_list_data["selected_utilization_kind"],
        "task_set_list": []
    }
    _log_lines = []

    for it in range(N):
        # Select task-set level parameters
        number_of_cpu = random.randint(number_of_cpu_range[0], number_of_cpu_range[1])

        # Distribute total utilization across CPUs.
        if utilization_is_total:
            if uniform_cpu_utilization:
                U_per_cpu_list = [U_selected / number_of_cpu] * number_of_cpu
            else:
                U_per_cpu_list = UUniFast(number_of_cpu, U_selected)
        else:
            U_per_cpu_list = [U_selected] * number_of_cpu

        U_cpu = U_selected / number_of_cpu if utilization_is_total else U_selected  # representative value for logging

        _log(f"=== Iteration {it} ===")
        _log("[Parameters] Selected values")
        _log(f"  selected_U_total: {U_selected if utilization_is_total else U_selected * number_of_cpu}")
        _log(f"  selected_U_cpu: {U_cpu}")
        _log(f"  number_of_cpu: {number_of_cpu}")
        _log(f"  U_per_cpu_list: {U_per_cpu_list}")

        # UUnifast
        per_cpu_task_U_list = {}
        task_set = {}
        for cpu in range(number_of_cpu):
            # Per-CPU task count and target utilizations
            U_cpu_i = U_per_cpu_list[cpu]
            number_of_tasks_per_cpu = random.randint(number_of_tasks_per_cpu_range[0], number_of_tasks_per_cpu_range[1])
            if uniform_task_utilization:
                task_U_list = [U_cpu_i / number_of_tasks_per_cpu] * number_of_tasks_per_cpu
            else:
                task_U_list = UUniFast(number_of_tasks_per_cpu, U_cpu_i)
            per_cpu_task_U_list[cpu] = task_U_list
            task_set[cpu] = []
            _log(f"  cpu {cpu} number_of_tasks_per_cpu: {number_of_tasks_per_cpu}")
            _log(f"  cpu {cpu} task_U_list: {task_U_list}")
        
        # Create taskset
        G_utilization_threshold = random.uniform(G_utilization_threshold_range[0], G_utilization_threshold_range[1])
        _log(f"  G_utilization_threshold: {G_utilization_threshold}")
        
        id = 0
        G_utilization = 0.0
        while True:
            skip_count = 0
            for cpu in range(number_of_cpu):
                if len(per_cpu_task_U_list[cpu]) == 0:
                    skip_count += 1
                    continue
                
                # Calculate utilization and period for this task
                target_U = per_cpu_task_U_list[cpu].pop(0)
                T = random.randint(period_range[0], period_range[1])
                T = int(T)
                D = T
                
                # Calculate C and G from the total execution budget.
                # G_ratio=0.5 -> C=50%, G=50%; G_ratio=0.8 -> C=20%, G=80%.
                G_plus_C = target_U * T
                G_ratio = random.uniform(G_ratio_range[0], G_ratio_range[1])
                total_work = max(1, int(round(G_plus_C)))
                G = int(round(total_work * G_ratio))
                C = total_work - G
                
                # Create segment list
                segment_list = []                
                
                # Otherwise: build inference segments
                number_of_inference_segments = random.randint(
                    number_of_inference_segments_range[0],
                    number_of_inference_segments_range[1],
                )

                # CPU-only task (no GPU segment) should be generated safely.
                if G <= 0 or number_of_inference_segments <= 0:
                    segment_list.append({
                        'C': C,
                        'G_segment': 0,
                        'per_splitting_overhead': per_splitting_overhead,
                        'max_block_count': 1,
                    })
                else:
                    C_list = split_int(C, number_of_inference_segments + 1)
                    G_list = split_int(G, number_of_inference_segments)

                    # Segment list
                    for i in range(number_of_inference_segments):
                        G_segment = max(1, G_list[i])
                        requested_max_blocks = random.randint(
                            max_block_count_range[0],
                            max_block_count_range[1],
                        )
                        feasible_max_blocks = max(1, G_segment)
                        max_block_count = min(requested_max_blocks, feasible_max_blocks)
                        segment_list.append({
                            'C': C_list[i],
                            'G_segment': G_segment,
                            'per_splitting_overhead': per_splitting_overhead,
                            'max_block_count': max_block_count,
                        })

                    # Last execution segment (CPU-only)
                    segment_list.append({
                        'C': C_list[number_of_inference_segments],
                        'G_segment': 0,
                        'per_splitting_overhead': per_splitting_overhead,
                        'max_block_count': 1,
                    })
                
                task_set[cpu].append(SegInfTask(id, segment_list, T, D, 1 / T, cpu=cpu))
                id += 1
                G_utilization += G / T
                
            if skip_count == number_of_cpu: break
        
        # Logging generated task set and actual values
        _log("")
        _log("[Task Set] Generated tasks")
        for cpu in task_set:
            _log(f"  cpu {cpu} tasks:")
            for task in task_set[cpu]:
                _log(f"    {task}")
            cpu_util = sum(task.C / task.T for task in task_set[cpu])
            cpu_c_plus_g_hat_util = sum((task.C + task.G) / task.T for task in task_set[cpu])
            _log(f"  cpu {cpu} C utilization: {cpu_util}")
            _log(f"  cpu {cpu} C+G utilization: {cpu_c_plus_g_hat_util}")
        
        _log("")
        _log("[Actual Values] Computed metrics")
        total_g_util = 0.0
        for cpu in task_set:
            cpu_total_util = sum((task.C + task.G) / task.T for task in task_set[cpu])
            _log(f"  cpu {cpu} total C+G utilization: {cpu_total_util}")
            _log(f"  cpu {cpu} per-task utilization:")
            for task in task_set[cpu]:
                task_util = (task.C + task.G) / task.T
                task_g_util = task.G / task.T
                _log(f"    task_id={task.id} C+G_utilization={task_util} G_utilization={task_g_util}")
                total_g_util += task_g_util
        _log(f"  taskset_total_G_utilization: {total_g_util}")
        _log("")

        # Collect task set data for reuse
        taskset_entry = {
            "iteration": it,
            "selected_U_total": U_selected if utilization_is_total else U_selected * number_of_cpu,
            "selected_U_cpu": U_cpu,
            "number_of_cpu": number_of_cpu,
            "G_utilization_threshold": G_utilization_threshold,
            "cpus": {}
        }
        taskset_pickle_entry = {
            "iteration": it,
            "selected_U_total": U_selected if utilization_is_total else U_selected * number_of_cpu,
            "selected_U_cpu": U_cpu,
            "number_of_cpu": number_of_cpu,
            "G_utilization_threshold": G_utilization_threshold,
            "cpus": {}
        }
        for cpu in task_set:
            taskset_entry["cpus"][str(cpu)] = [
                {
                    "id": task.id,
                    "cpu": cpu,
                    "C": task.C,
                    "G": task.G,
                    "period": task.period,
                    "deadline": task.deadline,
                    "priority": task.priority,
                    "C_list": task.C_list,
                    "G_segment_list": task.G_segment_list,
                    "max_block_count_list": task.max_block_count_list,
                    "m": task.m,
                }
                for task in task_set[cpu]
            ]
            taskset_pickle_entry["cpus"][cpu] = list(task_set[cpu])
        task_set_list_data["task_set_list"].append(taskset_entry)
        task_set_list_pickle_data["task_set_list"].append(taskset_pickle_entry)

    util_tag = f"{U_selected:.10f}".rstrip("0").rstrip(".")
    log_path = os.path.join(OUTPUT_DIR, f"simulation_log_u{util_tag}.txt")
    json_path = os.path.join(OUTPUT_DIR, f"task_set_list_u{util_tag}.json")
    pickle_path = os.path.join(OUTPUT_DIR, f"task_set_list_u{util_tag}.pkl")

    """
    Save log files for generated task set
    """
    with open(log_path, "w") as f:
        for line in _log_lines:
            f.write(f"{line}\n")

    with open(json_path, "w") as f:
        json.dump(task_set_list_data, f, indent=2)

    with open(pickle_path, "wb") as f:
        pickle.dump(task_set_list_pickle_data, f)
