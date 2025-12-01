from baseline_botorch_qlogei import run_botorch_qlogei
import numpy as np
import os
from baseline_plot import plot_problem
from baseline_postprocess import aggregate_problem


def run_baseline_all(funs=None, instances=None, dims=None, reps=5, base_outdir="results/baseline_qlogei", seed=0):
    """Run the standard baseline suite and postprocess results."""
    if funs is None:
        funs = [2, 4, 6, 50, 52, 54]
    if instances is None:
        instances = [0, 1, 2]
    if dims is None:
        dims = [2, 10]

    os.makedirs(base_outdir, exist_ok=True)
    for f in funs:
        for inst in instances:
            for d in dims:
                budget = 10 * d
                problem_dir = os.path.join(base_outdir, f"f{f}_i{inst}_d{d}")
                os.makedirs(problem_dir, exist_ok=True)
                print(f"Running baseline qLogEI: f{f}, inst {inst}, dim {d}, budget {budget}, reps {reps}")

                for rep in range(reps):
                    print(f"  Rep {rep}")
                    y_obj, y_con, X = run_botorch_qlogei(dim=d, budget=budget, coco_fun=f, coco_instance=inst, seed=seed + rep)

                    try:
                        y_obj_np = y_obj.detach().cpu().numpy()
                    except Exception:
                        y_obj_np = np.asarray(y_obj)
                    try:
                        y_con_np = y_con.detach().cpu().numpy()
                    except Exception:
                        y_con_np = np.asarray(y_con)

                    np.save(os.path.join(problem_dir, f"obj_rep{rep}.npy"), y_obj_np)
                    np.save(os.path.join(problem_dir, f"cons_rep{rep}.npy"), y_con_np)
                    np.save(os.path.join(problem_dir, f"X_rep{rep}.npy"), X)

                    try:
                        import torch
                        event = {'batch': {'Y': torch.as_tensor(y_obj_np), 'C': torch.as_tensor(y_con_np)}}
                        torch_fname = f"baseline_f{f}_i{inst}_d{d}_it_{rep}.torch"
                        torch.save([event], os.path.join(problem_dir, torch_fname))
                    except Exception:
                        pass

                print("  Postprocessing and plotting", problem_dir)
                try:
                    aggregate_problem(problem_dir, n_iteration=budget)
                except Exception as e:
                    print("  Aggregation failed:", e)
                try:
                    plot_problem(problem_dir, out_dir=problem_dir, show=False)
                except Exception as e:
                    print("  Plotting failed:", e)
                print("Saved and plotted:", problem_dir)
