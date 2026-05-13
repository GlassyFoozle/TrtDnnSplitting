import math
import os
from copy import deepcopy
from src.rta.task import InferenceSegment

NUMERATOR_EXPLOSION_LIMIT = 10**18
FLOAT_TIE_EPSILON = 1e-12
_SS_R_CHOICE_STATS = None


class NumeratorExplosionError(RuntimeError):
    pass


def reset_SS_R_choice_stats():
    global _SS_R_CHOICE_STATS
    _SS_R_CHOICE_STATS = {"req": 0, "job": 0}


def get_SS_R_choice_stats():
    stats = _SS_R_CHOICE_STATS or {"req": 0, "job": 0}
    req_count = int(stats.get("req", 0))
    job_count = int(stats.get("job", 0))
    return {
        "req": req_count,
        "job": job_count,
        "total": req_count + job_count,
    }


def _record_SS_R_choice(choice):
    if _SS_R_CHOICE_STATS is not None:
        _SS_R_CHOICE_STATS[choice] = _SS_R_CHOICE_STATS.get(choice, 0) + 1


def sort_task_set(task_set):
    cpus = task_set.get("cpus", {}) if isinstance(task_set, dict) and "cpus" in task_set else task_set
    tasks = []
    if isinstance(cpus, dict):
        for _, cpu_tasks in cpus.items():
            for task in cpu_tasks:
                tasks.append(task)
    return sorted(tasks, key=lambda t: (-t.priority, t.id))


def ceil_div_with_context(numerator, denominator, where, **context):
    ctx_text = " ".join(f"{k}={v}" for k, v in context.items())

    if denominator == 0:
        raise ZeroDivisionError(
            f"[{os.getpid()}] Zero denominator in {where}: "
            f"numerator={numerator} denominator={denominator} {ctx_text}"
        )

    if abs(numerator) > NUMERATOR_EXPLOSION_LIMIT:
        raise NumeratorExplosionError(
            f"[{os.getpid()}] Numerator explosion in {where}: "
            f"numerator={numerator} denominator={denominator} "
            f"limit={NUMERATOR_EXPLOSION_LIMIT} {ctx_text}"
        )

    return -(-numerator // denominator)


def get_max_lower_blocking(sorted_task_list, i):
    lp_tasks = sorted_task_list[i + 1:]
    if not lp_tasks:
        return 0
    return max(task_l.max_G_block for task_l in lp_tasks)


def get_B_i_req(sorted_task_list, i, R_list):
    task_i = sorted_task_list[i]
    max_lower_blocking = get_max_lower_blocking(sorted_task_list, i)

    B_i = 0
    for _ in range(max(task_i.m - 1, 0)):
        prev_B_i_j = max_lower_blocking
        while True:
            B_i_j = max_lower_blocking
            for h in range(i):
                task_h = sorted_task_list[h]
                numerator = prev_B_i_j + R_list[h] - task_h.C
                jobs = ceil_div_with_context(
                    numerator, task_h.T, "get_B_i_req",
                    task_i_idx=i, task_h_idx=h,
                    task_i_id=task_i.id, task_h_id=task_h.id,
                    prev_B_i_j=prev_B_i_j, R_h=R_list[h],
                    C_h=task_h.C, T_h=task_h.T,
                )
                B_i_j += jobs * task_h.G

            if B_i_j == prev_B_i_j:
                break
            prev_B_i_j = B_i_j

        B_i += B_i_j

    return B_i


def get_SS_R_req(sorted_task_list, i, R_list):
    task_i = sorted_task_list[i]
    max_lower_blocking = get_max_lower_blocking(sorted_task_list, i)

    B_i = get_B_i_req(sorted_task_list, i, R_list)

    initial_R_i = task_i.C + task_i.G
    prev_R_i = initial_R_i
    while True:
        R_i = task_i.C + task_i.G + B_i
        for h in range(i):
            task_h = sorted_task_list[h]
            if task_h.cpu != task_i.cpu:
                continue
            numerator = prev_R_i + R_list[h] - task_h.C
            jobs = ceil_div_with_context(
                numerator, task_h.T, "get_SS_R_req",
                task_i_idx=i, task_h_idx=h,
                task_i_id=task_i.id, task_h_id=task_h.id,
                prev_R_i=prev_R_i, R_h=R_list[h],
                C_h=task_h.C, T_h=task_h.T,
            )
            R_i += jobs * task_h.C

        if R_i == prev_R_i:
            break
        prev_R_i = R_i

    B_i_low = max(task_i.m - 1, 0) * max_lower_blocking
    B_i_high = B_i - B_i_low
    return R_i, B_i_high, B_i_low, initial_R_i


def get_SS_R_job(sorted_task_list, i, R_list):
    task_i = sorted_task_list[i]
    max_lower_blocking = get_max_lower_blocking(sorted_task_list, i)

    initial_R_i = task_i.C + task_i.G
    prev_R_i = initial_R_i
    while True:
        I_i = 0
        B_i = max(task_i.m - 1, 0) * max_lower_blocking
        for h in range(i):
            task_h = sorted_task_list[h]
            numerator = prev_R_i + R_list[h] - task_h.C
            jobs = ceil_div_with_context(
                numerator, task_h.T, "get_SS_R_job",
                task_i_idx=i, task_h_idx=h,
                task_i_id=task_i.id, task_h_id=task_h.id,
                prev_R_i=prev_R_i, R_h=R_list[h],
                C_h=task_h.C, T_h=task_h.T,
            )
            if task_h.cpu == task_i.cpu:
                I_i += jobs * task_h.C
            B_i += jobs * task_h.G

        R_i = task_i.C + task_i.G + B_i + I_i
        if R_i == prev_R_i:
            break
        prev_R_i = R_i

    B_i_low = max(task_i.m - 1, 0) * max_lower_blocking
    B_i_high = B_i - B_i_low
    return R_i, B_i_high, B_i_low, initial_R_i


def get_SS_R(sorted_task_list, i, R_list):
    task_i = sorted_task_list[i]
    R_req, R_req_B_high, R_req_B_low, _ = get_SS_R_req(sorted_task_list, i, R_list)
    R_job, R_job_B_high, R_job_B_low, _ = get_SS_R_job(sorted_task_list, i, R_list)

    if R_req <= R_job + FLOAT_TIE_EPSILON:
        R, B_high, B_low = R_req, R_req_B_high, R_req_B_low
        _record_SS_R_choice("req")
    else:
        R, B_high, B_low = R_job, R_job_B_high, R_job_B_low
        _record_SS_R_choice("job")

    I_i = R - task_i.C - task_i.G - B_high - B_low
    return R, B_high, B_low, I_i


def get_SS_tolerance(task, D, C, G, I, B_high):
    if task.m <= 1:
        return math.inf
    
    return (D - (C + G + I + B_high)) / (task.m - 1) # lower blocking은 GPU segment 갯수만큼 발생 가능하므로!

def find_splitting_target(sorted_task_list, trigger_task_index, target_tolerance):
    target_max_block = None
    target_task_index = None
    target_segment_index = None
    target_n = None

    for task_index in range(trigger_task_index + 1, len(sorted_task_list)):
        task = sorted_task_list[task_index]
        for segment_index, segment in enumerate(task.inference_segment_list):
            if segment.size >= segment.max_block_count:
                continue

            cur_max_block = max(segment.G_block_list) if segment.G_block_list else 0
            if cur_max_block <= target_tolerance:
                continue

            if target_max_block is None or cur_max_block > target_max_block:
                target_max_block = cur_max_block
                target_task_index = task_index
                target_segment_index = segment_index
                target_n = segment.size

    if target_max_block is None:
        return None

    return (target_task_index, target_segment_index, target_n)

def split_largest_block_excluding_highest(sorted_task_list, excluded_task_indices=None):
    if len(sorted_task_list) <= 1:
        return False, None

    excluded = set(excluded_task_indices or [])
    best = None
    for task_idx in range(1, len(sorted_task_list)):
        if task_idx in excluded:
            continue
        task = sorted_task_list[task_idx]
        for segment_idx, segment in enumerate(task.inference_segment_list):
            max_n = segment.max_block_count
            if segment.size >= max_n:
                continue

            current_segment_max_block = max(segment.G_block_list) if segment.G_block_list else 0
            candidate = (current_segment_max_block, -task_idx, -segment_idx)
            if best is None or candidate > best[0]:
                best = (candidate, task_idx, segment_idx, segment.size)

    if best is None:
        return False, None

    _, task_idx, segment_idx, current_n = best
    changed = sorted_task_list[task_idx].split_segment(segment_idx, current_n + 1)
    if not changed:
        return False, None
    return True, task_idx

def does_all_lower_meet_tolerance(sorted_task_list, trigger_task_index, target_tolerance):
    for task_l in sorted_task_list[trigger_task_index + 1:]:
        if task_l.max_G_block > target_tolerance:
            return False
    return True

def update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx):
    new_R_list = []
    new_tolerance_list = []
    for k in range(last_task_idx+1):
        task_k = sorted_task_list[k]
        R_k, B_k_high, _, I_k = get_SS_R(sorted_task_list, k, new_R_list)
        new_R_list.append(R_k)
                        
        D_k = task_k.D
        C_k = task_k.C
        G_k = task_k.G
        
        new_tolerance_list.append(get_SS_tolerance(task_k, D_k, C_k, G_k, I_k, B_k_high))
    return new_R_list, new_tolerance_list

def update_UNI_R_list_and_tolerance_list(sorted_task_list, last_task_idx):
    new_R_list = []
    new_tolerance_list = []
    for k in range(last_task_idx+1):
        task_k = sorted_task_list[k]
        R_k, K_k = get_UNI_R_and_K(sorted_task_list, k)
        new_R_list.append(R_k)
                        
        D_k = task_k.D
        C_k = task_k.C
        G_k = task_k.G
        
        new_tolerance_list.append(get_UNI_tolerance(sorted_task_list, k, K_k))
    return new_R_list, new_tolerance_list

def get_optimistic_SS_R(sorted_task_list):
    # Get max block when full split
    full_split_block_list = []
    copied_task_list = deepcopy(sorted_task_list)
    for task in copied_task_list:
        probe_task = deepcopy(task)
        for segment_idx in range(probe_task.m - 1):
            seg = probe_task.inference_segment_list[segment_idx]
            probe_task.split_segment(segment_idx, seg.max_block_count)
        full_split_block = probe_task.max_G_block
        full_split_block_list.append(full_split_block)
    
    # Calculate optimistic R
    optimistic_R_list = []
    for i in range(len(copied_task_list)):
        changed_lower_tasks = []
        for j in range(i + 1, len(copied_task_list)):
            task_j = copied_task_list[j]
            old_max_G_block = task_j.max_G_block
            new_max_G_block = full_split_block_list[j]
            if old_max_G_block == new_max_G_block:
                continue
            task_j.max_G_block = new_max_G_block
            changed_lower_tasks.append((task_j, old_max_G_block))
        try:
            R_i, _, _, _ = get_SS_R(copied_task_list, i, optimistic_R_list)
        finally:
            for task_j, old_max_G_block in changed_lower_tasks:
                task_j.max_G_block = old_max_G_block

        optimistic_R_list.append(R_i)
        
    return optimistic_R_list


def get_optimistic_UNI_R(sorted_task_list):
    # Get max block when full split
    full_split_block_list = []
    copied_task_list = deepcopy(sorted_task_list)
    for task in copied_task_list:
        probe_task = deepcopy(task)
        for segment_idx in range(probe_task.m - 1):
            seg = probe_task.inference_segment_list[segment_idx]
            probe_task.split_segment(segment_idx, seg.max_block_count)
        full_split_block = probe_task.max_G_block
        full_split_block_list.append(full_split_block)
    
    # Calculate optimistic R
    optimistic_R_list = []
    for i in range(len(copied_task_list)):
        changed_lower_tasks = []
        for j in range(i + 1, len(copied_task_list)):
            task_j = copied_task_list[j]
            old_max_G_block = task_j.max_G_block
            new_max_G_block = full_split_block_list[j]
            if old_max_G_block == new_max_G_block:
                continue
            task_j.max_G_block = new_max_G_block
            changed_lower_tasks.append((task_j, old_max_G_block))
        try:
            R_i, _ = get_UNI_R_and_K(copied_task_list, i)
        finally:
            for task_j, old_max_G_block in changed_lower_tasks:
                task_j.max_G_block = old_max_G_block

        optimistic_R_list.append(R_i)
        
    return optimistic_R_list

def convert_task_SS_to_UNI(task):
    task.convert_SS_to_UNI()
    return task

def convert_task_UNI_to_SS(task):
    task.convert_UNI_to_SS()
    return task

def convert_task_list_to_UNI(task_list):
    output_task_list = []
    for task in task_list:
        output_task_list.append(convert_task_SS_to_UNI(task))
    return output_task_list

def convert_task_list_to_SS(task_list):
    output_task_list = []
    for task in task_list:
        output_task_list.append(convert_task_UNI_to_SS(task))
    return output_task_list


def get_UNI_last_segment(task):
    if not getattr(task, "_UNI", False):
        print("get_UNI_last_segment(): Invalid task")
        exit()

    return task.inference_segment_list[0].G_block_list[-1]

def get_UNI_R_and_K(sorted_task_list, i):
    task_i = sorted_task_list[i]
    T_i = task_i.T
    C_i = task_i.C + task_i.G
    C_i_last = get_UNI_last_segment(task_i)

    # Busy period
    B_i = get_max_lower_blocking(sorted_task_list, i)
    
    I_i_prev = B_i + C_i
    I_i = 0
    while True:
        I_i = B_i
        for h in range(i + 1): # Include task i itself
            task_h = sorted_task_list[h]
            C_h = task_h.C + task_h.G
            T_h = task_h.T
            jobs = ceil_div_with_context(
                I_i_prev, T_h, "get_UNI_R_and_K_busy_period",
                task_i_idx=i, task_h_idx=h,
                task_i_id=task_i.id, task_h_id=task_h.id,
                I_i_prev=I_i_prev, T_h=T_h,
            )
            I_i += jobs * C_h
            
        if I_i == I_i_prev: break        
        I_i_prev = I_i

    K_i = ceil_div_with_context(
        I_i, T_i, "get_UNI_R_and_K_K_i",
        task_i_idx=i, task_i_id=task_i.id,
        I_i=I_i, T_i=T_i,
    )
    
    # start time
    s_i_k_list = []
    for k in range(1, K_i + 1):            
        s_i_k_prev = B_i + C_i - C_i_last
        for h in range(i):
            task_h = sorted_task_list[h]
            C_h = task_h.C + task_h.G
            s_i_k_prev += C_h

        s_i_k = 0
        while True:
            s_i_k = B_i + k * C_i - C_i_last
            for h in range(i):
                task_h = sorted_task_list[h]
                C_h = task_h.C + task_h.G
                T_h = task_h.T
                s_i_k += (math.floor(s_i_k_prev / T_h) + 1) * C_h

            if s_i_k == s_i_k_prev:
                break
            s_i_k_prev = s_i_k

        s_i_k_list.append(s_i_k)

    # response time
    f_i_k_list = [s_i_k + C_i_last for s_i_k in s_i_k_list]

    R_i_candidates = []
    for _k in range(K_i):
        k = _k + 1
        f_i_k = f_i_k_list[_k]
        R_i_candidates.append(f_i_k - (k - 1) * T_i)
    R_i = max(R_i_candidates)
    
    # Return R_i and K_i
    return R_i, K_i

def get_UNI_tolerance(sorted_task_list, i, K_i):
    task_i = sorted_task_list[i]
    T_i = task_i.T
    C_i = task_i.C + task_i.G
    C_i_last = get_UNI_last_segment(task_i)
    D_i = task_i.D
    
    tolerance_i = math.inf
    for k in range(1, K_i + 1):
        t_min = (k - 1) * T_i
        t_max = (k - 1) * T_i + D_i - C_i_last

        # refer Eq. (4) in Aromolo. et. al.
        t_candidates = []
        for j in range(i): # higher priorities
            task_j = sorted_task_list[j]
            h = 1 
            while True:
                candidate = h * task_j.T                
                if candidate < t_min:
                    h += 1
                    continue            
                
                if candidate > t_max:
                    break    
                
                t_candidates.append(candidate)
                h += 1

        t_candidates.append(t_max)

        tolerance_i_k_list = []
        for t in t_candidates:
            tolerance_i_k = t - k * C_i + C_i_last
            for h in range(i): # hihger priorities
                task_h = sorted_task_list[h]
                C_h = task_h.C + task_h.G
                T_h = task_h.T
                tolerance_i_k -= (math.floor(t / T_h) + 1) * C_h
            tolerance_i_k_list.append(tolerance_i_k)
        tolerance_i_k = max(tolerance_i_k_list)
        tolerance_i = min(tolerance_i, tolerance_i_k)

    return tolerance_i

def split_by_config(task, splitting_config):
    # Check validity
    segment = task.inference_segment_list[0]
    fixed = getattr(segment, "fixed_one_indices", set())
    if len(splitting_config) != max(segment.max_block_count - 1, 0):
        return None
    
    return task.split_by_config(0, splitting_config)

def add_split_point(cur_splitting_config, base_splitting_config, seen_splitting_configs, splitting_config_candidates):
    for config_idx, value in enumerate(cur_splitting_config):
        if base_splitting_config[config_idx] == 1:
            continue
        if value != 0:
            continue

        next_splitting_config = list(cur_splitting_config)
        next_splitting_config[config_idx] = 1
        next_key = tuple(next_splitting_config)

        if next_key in seen_splitting_configs:
            continue

        seen_splitting_configs.add(next_key)
        splitting_config_candidates.append(next_splitting_config)

def add_SS_split_point(cur_SS_splitting_config, seen_SS_splitting_config, SS_splitting_config_candidates):
    for segment_idx, current_config in enumerate(cur_SS_splitting_config):        
        for config_idx, value in enumerate(current_config):            
            if value != 0:
                continue
            next_SS_splitting_config = copy_SS_splitting_config(cur_SS_splitting_config)
            next_SS_splitting_config[segment_idx][config_idx] = 1
            key = get_SS_splitting_config_key(next_SS_splitting_config)
            if key in seen_SS_splitting_config:
                continue
            seen_SS_splitting_config.add(key)
            SS_splitting_config_candidates.append(next_SS_splitting_config)
    
    return

def get_SS_splitting_config(task):
    SS_splitting_config = []
    for segment in task.inference_segment_list:
        SS_splitting_config.append(segment.splitting_config)
    return SS_splitting_config

def copy_SS_splitting_config(SS_splitting_config):
    return [list(config) for config in SS_splitting_config]

def get_SS_splitting_config_key(configs):
    return tuple(tuple(config) for config in configs)

def apply_SS_splitting_config(task, SS_splitting_config):
    if len(SS_splitting_config) != len(task.inference_segment_list):
        return False
    for segment_idx, config in enumerate(SS_splitting_config):
        if not task.split_by_config(segment_idx, config):
            return False
    return True


def RTA_SS_single(task_set):
    sorted_task_list = sort_task_set(task_set)
    R_list = []

    # Analysis
    for i in range(len(sorted_task_list)):
        R_i, _, _, _ = get_SS_R(sorted_task_list, i, R_list)
        if R_i > sorted_task_list[i].D:
            return False
        R_list.append(R_i)

    return True

def RTA_SS_max(task_set):
    sorted_task_list = sort_task_set(task_set)

    # full splitting
    for task in sorted_task_list:
        for segment_idx in range(task.m - 1):
            segment = task.inference_segment_list[segment_idx]
            task.split_segment(segment_idx, segment.max_block_count)

    # Analysis
    R_list = []
    for i in range(len(sorted_task_list)):
        R_i, _, _, _ = get_SS_R(sorted_task_list, i, R_list)
        if R_i > sorted_task_list[i].D:
            return False
        R_list.append(R_i)

    return True

def RTA_SS_tol(task_set):
    is_schedulable = True
    profiling_count = 0
    sorted_task_list = sort_task_set(task_set)
    
    R_list = []
    tolerance_list = [math.inf for _ in range(len(sorted_task_list))]    
    for i in range(len(sorted_task_list)):
        # Step 1: SS RTA
        profiling_count += 1
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D
        
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
                
        is_last_task = (i == len(sorted_task_list) - 1)
        
        # Update tolerance
        if not is_last_task:
            tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_i_high)            
        else:
            tolerance_i = math.inf            
        tolerance_list[i] = tolerance_i
                
        if R_i <= D_i: # Current task meets deadline
            R_list.append(R_i)
            continue
        
        # Step 2: tolerance-fit splitting
        target_tolerance = min(tolerance_list[: i+1])
        if is_last_task or tolerance_i <= 0:            
            is_schedulable = False
            return is_schedulable, profiling_count        

        meet_tolerance = False
        while True:
            target_tolerance = min(tolerance_list[: i+1])
            
            # All lower tasks meet tolerance
            if does_all_lower_meet_tolerance(sorted_task_list,i,target_tolerance):
                meet_tolerance = True
                break
            
            # Find splitting target
            split_target = find_splitting_target(sorted_task_list, i, target_tolerance)

            # No more splitting target                                            
            if split_target is None: 
                meet_tolerance = False
                break

            # Apply splitting
            target_task_index, target_segment_index, current_n = split_target
            is_split = sorted_task_list[target_task_index].split_segment(target_segment_index,current_n + 1)
            
            # Not splitable
            if not is_split:
                meet_tolerance = False
                break
            
            profiling_count += 1
            
            # Update R_list and tolerance_list
            R_list, new_tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
            tolerance_list[: i + 1] = new_tolerance_list
        
        if not meet_tolerance:
            is_schedulable = False            
        else:
            if len(R_list) <= i:
                R_list.append(R_i)
        
        # Detect not schedulable
        if not is_schedulable:
            break
    
    return is_schedulable, profiling_count

def _RTA_SS_tol_fb_impl(task_set, early_stop=False):
    is_schedulable = True
    profiling_count = 0
    sorted_task_list = sort_task_set(task_set)
    
    R_list = []
    tolerance_list = [math.inf for _ in range(len(sorted_task_list))]    
    
    i = 0
    while i < len(sorted_task_list):
        '''
        # Step 1: SS RTA #####
        '''
        profiling_count += 1
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D
        
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
                
        is_last_task = (i == len(sorted_task_list) - 1)
        
        # Update tolerance
        if not is_last_task:
            tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_i_high)            
        else:
            tolerance_i = math.inf            
        tolerance_list[i] = tolerance_i
                
        if R_i <= D_i: # Current task meets deadline
            if len(R_list) <= i:
                R_list.append(R_i)
            else:
                R_list[i] = R_i
            i += 1
            continue
        
        '''
        # Step 2: tolerance-fit splitting
        '''

        meet_tolerance = False
        while True:
            # If invalid, go to fallback
            target_tolerance = min(tolerance_list[: i+1])
            if is_last_task or tolerance_i <= 0:            
                break   
            
            # All lower tasks meet tolerance
            if does_all_lower_meet_tolerance(sorted_task_list,i,target_tolerance):
                meet_tolerance = True
                break
            
            # Find splitting target
            split_target = find_splitting_target(sorted_task_list, i, target_tolerance)

            # No more splitting target                                            
            if split_target is None: 
                meet_tolerance = False
                break

            # Apply splitting
            target_task_index, target_segment_index, current_n = split_target
            is_split = sorted_task_list[target_task_index].split_segment(target_segment_index,current_n + 1)
            
            # Not splitable
            if not is_split:
                meet_tolerance = False
                break
            
            profiling_count += 1
            
            # Update R_list and tolerance_list
            R_list, new_tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
            tolerance_list[: i + 1] = new_tolerance_list
                    
        if meet_tolerance:
            if len(R_list) <= i:
                R_list.append(R_i)
            i += 1
            continue
        
        '''
        # Step 3: Fallback
        '''
        is_splitted, splitted_task_idx = split_largest_block_excluding_highest(sorted_task_list)
        if not is_splitted or splitted_task_idx is None:
            is_schedulable = False
            break 
        profiling_count += 1
        
        # Update task idx if splited task idx is higher
        restart_idx = i if i <= splitted_task_idx else splitted_task_idx
        
        # Update R_list and tolerance_list
        if R_list:
            R_list, new_tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=(len(R_list)-1))
            tolerance_list[: len(new_tolerance_list)] = new_tolerance_list
        else:
            R_list = []
        
        '''
        # Step 4: Early stop
        '''
        if early_stop:
            optimistic_R_list = get_optimistic_SS_R(sorted_task_list)
            for k in range(len(optimistic_R_list)):
                task_k = sorted_task_list[k]
                D_k = task_k.D
                R_k = optimistic_R_list[k]
                if D_k < R_k:
                    is_schedulable = False
                    return is_schedulable, profiling_count
        
        # Detect not schedulable
        if not is_schedulable:
            break
        
        # Reset target index
        i = restart_idx
        
    return is_schedulable, profiling_count

def RTA_SS_tol_fb(task_set):
    return _RTA_SS_tol_fb_impl(task_set, early_stop=False)

def RTA_SS_tol_fb_early(task_set):
    return _RTA_SS_tol_fb_impl(task_set, early_stop=True)


def RTA_UNI_tol_fb(task_set, early_stop=True):
    # Convert to unified resource model
    sorted_task_list = []
    for task in sort_task_set(task_set):
        sorted_task_list.append(convert_task_SS_to_UNI(task))
        
    is_schedulable = True
    profiling_count = 0    
    
    R_list = []
    tolerance_list = [math.inf for _ in range(len(sorted_task_list))]    
    
    i = 0
    while i < len(sorted_task_list):
        '''
        # Step 1: UNI RTA
        '''
        profiling_count += 1
        
        task_i = sorted_task_list[i]
        D_i = task_i.D
        R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)
                        
        is_last_task = (i == len(sorted_task_list) - 1)
        
        # Update tolerance
        if not is_last_task:
            tolerance_i = get_UNI_tolerance(sorted_task_list, i, K_i)            
        else:
            tolerance_i = math.inf            
        tolerance_list[i] = tolerance_i
                
        if R_i <= D_i: # Current task meets deadline
            if len(R_list) <= i:
                R_list.append(R_i)
            else:
                R_list[i] = R_i
            i += 1
            continue
        
        '''
        # Step 2: tolerance-fit splitting
        '''

        meet_tolerance = False
        while True:
            # If invalid, go to fallback
            target_tolerance = min(tolerance_list[: i+1])
            if is_last_task or tolerance_i <= 0:            
                break   
            
            # All lower tasks meet tolerance            
            sorted_task_list = convert_task_list_to_SS(sorted_task_list)
            if does_all_lower_meet_tolerance(sorted_task_list,i,target_tolerance):
                meet_tolerance = True
                sorted_task_list = convert_task_list_to_UNI(sorted_task_list)
                break
            
            # Find splitting target
            split_target = find_splitting_target(sorted_task_list, i, target_tolerance)

            # No more splitting target                                            
            if split_target is None: 
                meet_tolerance = False
                sorted_task_list = convert_task_list_to_UNI(sorted_task_list)    
                break

            # Apply splitting
            target_task_index, target_segment_index, current_n = split_target
            is_split = sorted_task_list[target_task_index].split_segment(target_segment_index,current_n + 1)
            sorted_task_list = convert_task_list_to_UNI(sorted_task_list)
            
            # Not splitable
            if not is_split:
                meet_tolerance = False
                break
            
            profiling_count += 1
            
            # Update R_list and tolerance_list
            R_list, new_tolerance_list = update_UNI_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
            tolerance_list[: i + 1] = new_tolerance_list
                    
        if meet_tolerance:
            if len(R_list) <= i:
                R_list.append(R_i)
            i += 1
            continue
        
        '''
        # Step 3: Fallback
        '''
        sorted_task_list = convert_task_list_to_SS(sorted_task_list)
        is_splitted, splitted_task_idx = split_largest_block_excluding_highest(sorted_task_list)
        sorted_task_list = convert_task_list_to_UNI(sorted_task_list)
        if not is_splitted or splitted_task_idx is None:
            is_schedulable = False
            break 
        profiling_count += 1        
        
        # Update task idx if splited task idx is higher
        restart_idx = i if i <= splitted_task_idx else splitted_task_idx
        
        # Update R_list and tolerance_list
        if R_list:
            R_list, new_tolerance_list = update_UNI_R_list_and_tolerance_list(sorted_task_list, last_task_idx=(len(R_list)-1))
            tolerance_list[: len(new_tolerance_list)] = new_tolerance_list
        else:
            R_list = []
        
        '''
        # Step 4: Early stop
        '''
        if early_stop:
            optimistic_R_list = get_optimistic_UNI_R(sorted_task_list)
            for k in range(len(optimistic_R_list)):
                task_k = sorted_task_list[k]
                D_k = task_k.D
                R_k = optimistic_R_list[k]
                if D_k < R_k:
                    is_schedulable = False
                    return is_schedulable, profiling_count
        
        # Detect not schedulable
        if not is_schedulable:
            break
        
        # Reset target index
        i = restart_idx
        
    return is_schedulable, profiling_count
    
    
    

def RTA_UNI_opt(task_set):
    # Convert to unified resource model
    sorted_task_list = []
    for task in sort_task_set(task_set):
        sorted_task_list.append(convert_task_SS_to_UNI(task))
    
    is_schedulable = True
    tolerance_list = []
    profiling_count = len(sorted_task_list)
    
    '''
    Step 1: Calculate R of task 0
    '''
    i = 0
    task_i = sorted_task_list[i]        
        
    R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)
    tolerance_i = get_UNI_tolerance(sorted_task_list, i, K_i)
    tolerance_list.append(tolerance_i)
    
    '''
    Step 2: Calculate R of task 1 ... n-1 in priority order
    '''
    for i in range(1, len(sorted_task_list)):
        task_i = sorted_task_list[i]
        cur_tolerance = min(tolerance_list)
                                
        # Filter full split        
        task_i_full_split = deepcopy(task_i)
        if not task_i_full_split.split_all_segments():
            is_schedulable = False
            return is_schedulable, profiling_count
        profiling_count += 1
                
        if cur_tolerance < task_i_full_split.max_G_block:
            is_schedulable = False
            return is_schedulable, profiling_count
        
        # splitting config set
        # NOTE: non_splitting_config cand include 1s for C segments
        non_splitting_config = task_i.non_splitting_config
        splitting_config_candidates = [non_splitting_config]
        seen_splitting_configs = {tuple(non_splitting_config)}
        
        # splitting
        minimal_WCET = math.inf
        selected_splitting_config = non_splitting_config
        while len(splitting_config_candidates) > 0:
            cur_splitting_config = splitting_config_candidates.pop(0)
            result = split_by_config(task_i, cur_splitting_config)
            if result is None:
                print("[ERROR] Invalid split by config in RTA_UNI_opt")
                is_schedualble = False
                return is_schedulable, profiling_count
            if result is False: # Cannot split
                continue
                        
            profiling_count += 1
            
            # split more
            if cur_tolerance < task_i.max_G_block:
                add_split_point(
                    cur_splitting_config,
                    non_splitting_config,
                    seen_splitting_configs,
                    splitting_config_candidates,
                )
            else:
                C_i = task_i.C + task_i.G
                if C_i < minimal_WCET:
                    minimal_WCET = C_i
                    selected_splitting_config = cur_splitting_config
        
        # Apply selected splitting
        if split_by_config(task_i, selected_splitting_config) is None:
            print("[ERROR] Invalid split by seleceted config in RTA_UNI_opt")
            is_schedualble = False
            return is_schedulable, profiling_count
                
        # Update tolerance
        R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)
        tolerance_i = get_UNI_tolerance(sorted_task_list, i, K_i)
        tolerance_list.append(tolerance_i)
    
    # Final check
    for i in range(len(sorted_task_list)):
        task_i = sorted_task_list[i]
        D_i = task_i.D
        R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)

        # schedulability check
        if R_i > D_i:
            is_schedulable = False            
            return is_schedulable, profiling_count
        
    return is_schedulable, profiling_count


def RTA_UNI_heu(task_set):
    # Convert to unified resource model
    sorted_task_list = []
    for task in sort_task_set(task_set):
        sorted_task_list.append(convert_task_SS_to_UNI(task))
    
    is_schedulable = True
    tolerance_list = []
    profiling_count = len(sorted_task_list)
    
    '''
    Step 1: Calculate R of task 0
    '''
    i = 0
    task_i = sorted_task_list[i]        
        
    R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)
    tolerance_i = get_UNI_tolerance(sorted_task_list, i, K_i)
    tolerance_list.append(tolerance_i)
    
    '''
    Step 2: Calculate R of task 1 ... n-1 in priority order
    '''
    for i in range(1, len(sorted_task_list)):
        task_i = sorted_task_list[i]
        cur_tolerance = min(tolerance_list)
                        
        # Filter full split                
        task_i_full_split = deepcopy(task_i)
        if not task_i_full_split.split_all_segments():
            is_schedulable = False
            return is_schedulable, profiling_count
        profiling_count += 1
                
        if cur_tolerance < task_i_full_split.max_G_block:
            is_schedulable = False
            return is_schedulable, profiling_count
        
        # splitting config set
        # NOTE: non_splitting_config cand include 1s for C segments
        non_splitting_config = task_i.non_splitting_config
        splitting_config_candidates = [non_splitting_config]
        seen_splitting_configs = {tuple(non_splitting_config)}
        
        selected_splitting_config = non_splitting_config
        
        # split more
        while True: 
            if cur_tolerance >= task_i.max_G_block:
                break
            
            add_split_point(
                selected_splitting_config,
                non_splitting_config,
                seen_splitting_configs,
                splitting_config_candidates,
            )
            
            if len(splitting_config_candidates) == 0: break
            
            minimal_max_block = math.inf
            while len(splitting_config_candidates) > 0:
                cur_splitting_config = splitting_config_candidates.pop(0)
                if split_by_config(task_i, cur_splitting_config) is None:
                    print("[ERROR] Invalid split by seleceted config in RTA_UNI_heu")
                    is_schedualble = False
                    return is_schedulable, profiling_count
                profiling_count += 1
                
                if task_i.max_G_block < minimal_max_block:
                    minimal_max_block = task_i.max_G_block
                    selected_splitting_config = cur_splitting_config
            
            if split_by_config(task_i, selected_splitting_config) is None:
                print("[ERROR] Invalid split by intermediate selected config in RTA_UNI_heu")
                is_schedualble = False
                return is_schedulable, profiling_count
                
        # Apply selected splitting
        if split_by_config(task_i, selected_splitting_config) is None:
            print("[ERROR] Invalid split by seleceted config in RTA_UNI_heu")
            is_schedualble = False
            return is_schedulable, profiling_count
                
        # Update tolerance
        R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)
        tolerance_i = get_UNI_tolerance(sorted_task_list, i, K_i)
        tolerance_list.append(tolerance_i)
    
    # Final check
    for i in range(len(sorted_task_list)):
        task_i = sorted_task_list[i]
        D_i = task_i.D
        R_i, K_i = get_UNI_R_and_K(sorted_task_list, i)

        # schedulability check
        if R_i > D_i:
            is_schedulable = False            
            return is_schedulable, profiling_count
        
    return is_schedulable, profiling_count

# NOTE: Update R and tolerance at the end of each for loop
def RTA_SS_opt(task_set):
    is_schedulable = True
    tolerance_list = []
    sorted_task_list = sort_task_set(task_set)
    profiling_count = len(sorted_task_list)
    R_list = []
    
    '''
    Step 1: Calculate R of task 0
    '''
    i = 0
    task_i = sorted_task_list[i]
    C_i = task_i.C
    G_i = task_i.G
    D_i = task_i.D
        
    R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
    tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_i_high)
    tolerance_list.append(tolerance_i)
    R_list.append(R_i)
    
    '''
    Step 2: Calculate R of task 1 ... n-1 in priority order
    '''
    for i in range(1, len(sorted_task_list)):
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D
        cur_tolerance = min(tolerance_list)
                                
        # Filter full split        
        task_i_full_split = deepcopy(task_i)
        if not task_i_full_split.split_all_segments():
            is_schedulable = False
            return is_schedulable, profiling_count
        profiling_count += 1
                
        if cur_tolerance < task_i_full_split.max_G_block:
            is_schedulable = False
            return is_schedulable, profiling_count
        
        # splitting config set
        # NOTE: SS_splitting_config: list of splitting configs in a task
        base_SS_splitting_config = get_SS_splitting_config(task_i)
                
        SS_splitting_config_candidates = [copy_SS_splitting_config(base_SS_splitting_config)]
        seen_SS_splitting_configs = {get_SS_splitting_config_key(base_SS_splitting_config)}
        
        # splitting
        minimal_WCET = math.inf
        selected_SS_splitting_config = base_SS_splitting_config
        while len(SS_splitting_config_candidates) > 0:
            cur_SS_splitting_config = SS_splitting_config_candidates.pop(0)
            if not apply_SS_splitting_config(task_i, cur_SS_splitting_config):
                continue
            profiling_count += 1
            
            # split more
            if cur_tolerance < task_i.max_G_block:
                add_SS_split_point(
                    cur_SS_splitting_config,
                    seen_SS_splitting_configs,
                    SS_splitting_config_candidates
                )                
            else:
                C_i = task_i.C + task_i.G
                if C_i < minimal_WCET:
                    minimal_WCET = C_i
                    selected_SS_splitting_config = cur_SS_splitting_config
        
        # Apply selected splitting
        if apply_SS_splitting_config(task_i, selected_SS_splitting_config) is None:
            print("[ERROR] Invalid split by seleceted config in RTA_UNI_opt")
            is_schedualble = False
            return is_schedulable, profiling_count
                                
        # Update R and tolerance
        R_list, tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
    
    # Final check
    for i in range(len(sorted_task_list)):
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
        
        # schedulability check
        if R_i > D_i:
            is_schedulable = False            
            return is_schedulable, profiling_count
        
    return is_schedulable, profiling_count

def RTA_SS_heu(task_set):
    is_schedulable = True
    tolerance_list = []
    sorted_task_list = sort_task_set(task_set)
    profiling_count = len(sorted_task_list)
    R_list = []
    
    '''
    Step 1: Calculate R of task 0
    '''
    i = 0
    task_i = sorted_task_list[i]
    C_i = task_i.C
    G_i = task_i.G
    D_i = task_i.D
        
    R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
    tolerance_i = get_SS_tolerance(task_i, D_i, C_i, G_i, I_i, B_i_high)
    tolerance_list.append(tolerance_i)
    R_list.append(R_i)
    
    '''
    Step 2: Calculate R of task 1 ... n-1 in priority order
    '''
    for i in range(1, len(sorted_task_list)):
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D
        cur_tolerance = min(tolerance_list)
                        
        # Filter full split        
        task_i_full_split = deepcopy(task_i)
        if not task_i_full_split.split_all_segments():
            is_schedulable = False
            return is_schedulable, profiling_count
        profiling_count += 1
                
        if cur_tolerance < task_i_full_split.max_G_block:
            is_schedulable = False
            return is_schedulable, profiling_count
        
        # splitting config set
        # NOTE: SS_splitting_config: list of splitting configs in a task        
        base_SS_splitting_config = get_SS_splitting_config(task_i)
                
        SS_splitting_config_candidates = [copy_SS_splitting_config(base_SS_splitting_config)]
        seen_SS_splitting_configs = {get_SS_splitting_config_key(base_SS_splitting_config)}
        
        # splitting
        minimal_WCET = math.inf
        selected_SS_splitting_config = base_SS_splitting_config
        
        # split more
        while True: 
            if cur_tolerance >= task_i.max_G_block:
                break
            
            add_SS_split_point(
                selected_SS_splitting_config,
                seen_SS_splitting_configs,
                SS_splitting_config_candidates
            )                
            
            if len(SS_splitting_config_candidates) == 0: break
            
            minimal_max_block = math.inf
            while len(SS_splitting_config_candidates) > 0:
                cur_SS_splitting_config = SS_splitting_config_candidates.pop(0)
                if not apply_SS_splitting_config(task_i, cur_SS_splitting_config):
                    continue
                profiling_count += 1
                
                if task_i.max_G_block < minimal_max_block:
                    minimal_max_block = task_i.max_G_block
                    selected_SS_splitting_config = cur_SS_splitting_config
            
            if not apply_SS_splitting_config(task_i, selected_SS_splitting_config):
                print("[ERROR] Invalid split by intermediate selected config in RTA_UNI_heu")
                is_schedualble = False
                return is_schedulable, profiling_count
                
        # Apply selected splitting
        if not apply_SS_splitting_config(task_i, selected_SS_splitting_config):
            print("[ERROR] Invalid split by seleceted config in RTA_SS_heu")
            is_schedualble = False
            return is_schedulable, profiling_count
                
        # Update tolerance
        R_list, tolerance_list = update_SS_R_list_and_tolerance_list(sorted_task_list, last_task_idx=i)
    
    # Final check
    for i in range(len(sorted_task_list)):
        task_i = sorted_task_list[i]
        C_i = task_i.C
        G_i = task_i.G
        D_i = task_i.D
        R_i, B_i_high, B_i_low, I_i = get_SS_R(sorted_task_list, i, R_list)
        
        # schedulability check
        if R_i > D_i:
            is_schedulable = False            
            return is_schedulable, profiling_count
        
    return is_schedulable, profiling_count
