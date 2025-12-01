import os
import numpy as np
import matplotlib.pyplot as plt


def plot_problem(problem_dir, out_dir=None, show=False):
    """Load aggregated numpy results and save a summary plot."""
    y_mono_path = os.path.join(problem_dir, "01_Y_mono.npy")
    y_best_path = os.path.join(problem_dir, "02_Y_best.npy")
    c_best_path = os.path.join(problem_dir, "02_C_best.npy")

    if not (os.path.exists(y_mono_path) and os.path.exists(y_best_path) and os.path.exists(c_best_path)):
        print("Missing processed npy files in", problem_dir)
        return

    y_best = np.load(y_best_path)
    if y_best.ndim == 1:
        y_best = y_best.reshape(1, -1)

    n_reps, n_evals = y_best.shape
    x = np.arange(1, n_evals + 1)

    fig, ax = plt.subplots()
    for i in range(n_reps):
        ax.plot(x, y_best[i], color='gray', alpha=0.25, linewidth=1)

    mean_curve = np.mean(y_best, axis=0)
    std_curve = np.std(y_best, axis=0)
    ax.plot(x, mean_curve, color='C0', linewidth=2.5, label='Mean best feasible')
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color='C0', alpha=0.2)

    ax.set_title(os.path.basename(problem_dir))
    ax.set_xlabel(f'Evaluation (T={n_evals})')
    ax.set_ylabel('Best feasible objective')
    ax.legend()

    if out_dir is None:
        out_dir = problem_dir
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "best_feasible.png"), bbox_inches='tight')
    plt.close(fig)
    print("Saved aggregated plot for", problem_dir)
