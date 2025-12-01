import os
import numpy as np
import torch


def aggregate_problem(problem_dir, n_iteration=None):
    """Aggregate per-repetition results in `problem_dir` and save summary npy files."""
    files = sorted(os.listdir(problem_dir))
    torch_files = [f for f in files if f.endswith('.torch')]

    states = []
    if torch_files:
        for tf in torch_files:
            path = os.path.join(problem_dir, tf)
            try:
                st = torch.load(path, map_location=torch.device('cpu'))
            except Exception:
                continue
            if isinstance(st, dict) or torch.is_tensor(st):
                st = [st]
            states.append(st)
    else:
        obj_files = sorted([f for f in files if f.startswith('obj_rep') and f.endswith('.npy')])
        for objf in obj_files:
            rep_idx = objf.replace('obj_rep', '').replace('.npy', '')
            consf = f'cons_rep{rep_idx}.npy'
            obj_path = os.path.join(problem_dir, objf)
            cons_path = os.path.join(problem_dir, consf)
            if not os.path.exists(cons_path):
                continue
            Y = np.load(obj_path)
            C = np.load(cons_path)
            Y_t = torch.as_tensor(Y)
            C_t = torch.as_tensor(C)
            event = {'batch': {'Y': Y_t, 'C': C_t}}
            states.append([event])

    if not states:
        raise RuntimeError(f'No repetition files found in {problem_dir}')

    if n_iteration is None:
        lengths = []
        for state in states:
            total = 0
            for ev in state:
                y = ev['batch']['Y']
                total += y.numel() if torch.is_tensor(y) else np.asarray(y).size
            lengths.append(total)
        n_iteration = min(lengths)

    Y_batch = []
    C_batch = []
    for state in states:
        Ys = []
        Cs = []
        for ev in state:
            y = ev['batch']['Y']
            c = ev['batch']['C']
            y_np = y.detach().cpu().numpy().reshape(-1) if torch.is_tensor(y) else np.asarray(y).reshape(-1)
            c_np = c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
            c_max = np.max(c_np, axis=1) if c_np.ndim == 2 else c_np.reshape(-1)
            Ys.append(y_np)
            Cs.append(c_max)

        Y_concat = np.concatenate(Ys, axis=0)[:n_iteration]
        C_concat = np.concatenate(Cs, axis=0)[:n_iteration]
        Y_batch.append(Y_concat)
        C_batch.append(C_concat)

    Y_best = np.array(Y_batch)
    C_best = np.array(C_batch)

    Y_f = np.copy(Y_best)
    C_f = np.copy(C_best)
    mask = (C_f > 0)
    if mask.any():
        max_per_row = np.max(Y_f, axis=1)
        for i in range(Y_f.shape[0]):
            Y_f[i, mask[i]] = max_per_row[i]

    Y_f_monotonic = []
    for YY in Y_f:
        y_mono = []
        for yy in YY:
            if not y_mono:
                y_mono = [yy]
            else:
                y_mono.append(yy if yy < y_mono[-1] else y_mono[-1])
        Y_f_monotonic.append(y_mono)

    Y_f_monotonic = np.array(Y_f_monotonic)
    np.save(os.path.join(problem_dir, '01_Y_mono.npy'), Y_f_monotonic)
    np.save(os.path.join(problem_dir, '02_Y_best.npy'), Y_best)
    np.save(os.path.join(problem_dir, '02_C_best.npy'), C_best)

    return Y_f_monotonic, Y_best, C_best
