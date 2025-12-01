import os
import numpy as np
from baseline_botorch_qlogei import run_botorch_qlogei
from baseline_plot import plot_problem
from baseline_postprocess import aggregate_problem

def to_numpy(arr):
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(arr, _np.ndarray):
            return arr
    except Exception:
        pass
    try:
        return np.asarray(arr)
    except Exception:
        return np.array([arr])


functions_to_run = [2, 4, 6, 50, 52, 54]      
instances_to_run = [0, 1, 2]                 
dimensions_to_run = [2, 10]            
repetitions_per_instance = 5
outdir = "results"
os.makedirs(outdir, exist_ok=True)

# --- Build task list ---
tasks = []
for f in functions_to_run:
    for inst in instances_to_run:
        for d in dimensions_to_run:
            budget = 10 * d
            problem_dir = os.path.join(
                outdir,
                f"bbob-constrained_f{f:03d}_i{inst + 1:02d}_d{d:02d}"
            )
            os.makedirs(problem_dir, exist_ok=True)
            for rep in range(repetitions_per_instance):
                tasks.append((f, inst, d, rep, budget, problem_dir))

# --- Execute tasks ---
for (f, inst, d, rep, budget, problem_dir) in tasks:
    print(f"Running qLogEI baseline: f{f}, inst {inst}, dim {d}, rep {rep}, budget {budget}")

    y_obj, y_con, X = run_botorch_qlogei(
        dim=d,
        budget=budget,
        coco_fun=f,
        coco_instance=inst,
        seed=rep,
    )

    y_obj_np = to_numpy(y_obj)
    y_con_np = None if y_con is None else to_numpy(y_con)
    X_np = to_numpy(X)

    np.save(os.path.join(problem_dir, f"obj_rep{rep}.npy"), y_obj_np)
    np.save(os.path.join(problem_dir, f"cons_rep{rep}.npy"), y_con_np)
    np.save(os.path.join(problem_dir, f"X_rep{rep}.npy"), X_np)


    try:
        import torch
        c_for_torch = y_con_np if y_con_np is not None else np.zeros_like(y_obj_np)
        event = {'batch': {'Y': torch.as_tensor(y_obj_np), 'C': torch.as_tensor(c_for_torch)}}
        torch_fname = f"baseline_f{f}_i{inst}_d{d}_it_{rep}.torch"
        torch.save([event], os.path.join(problem_dir, torch_fname))
    except Exception:
        pass

# --- Postprocess and plot ---
problem_dirs = sorted({t[5] for t in tasks})
for problem_dir in problem_dirs:
    try:
        aggregate_problem(problem_dir)
    except Exception:
        pass
    try:
        plot_problem(problem_dir, out_dir=problem_dir, show=True)
    except Exception:
        pass
