import random
import math
import numpy as np
import matplotlib.pyplot as plt


# Laboratorium 7: Ocena i wizualizacja rozwiązań wielokryterialnych
# Zadanie 1) SA + Pareto front + HVI dla kryteriów 3 i 4


def generate_instance(n_jobs, seed=None):
    """
    Generuje czasy przetwarzania p[i][j] i terminy d[j].
    """
    if seed is not None:
        random.seed(seed)
    p = [[random.randint(1, 99) for _ in range(n_jobs)] for _ in range(3)]
    S = sum(sum(row) for row in p)
    d = [random.randint(S//4, S//2) for _ in range(n_jobs)]
    return p, d


def compute_completion_times(pi, p):
    m, n = len(p), len(pi)
    C = np.zeros((m, n), dtype=int)
    C[0, 0] = p[0][pi[0]]
    for j in range(1, n):
        C[0, j] = C[0, j-1] + p[0][pi[j]]
    for i in range(1, m):
        C[i, 0] = C[i-1, 0] + p[i][pi[0]]
        for j in range(1, n):
            C[i, j] = max(C[i, j-1], C[i-1, j]) + p[i][pi[j]]
    return C


def objectives_two(C, d, pi):
    # Kryteria 3 (Tmax) i 4 (ΣT).
    tardiness = [max(int(C[-1, j] - d[pi[j]]), 0) for j in range(len(pi))]
    return max(tardiness), sum(tardiness)


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def pareto_front(objs):
    indices = []
    for i, oi in enumerate(objs):
        if not any(dominates(oj, oi) for j, oj in enumerate(objs) if j != i):
            indices.append(i)
    return indices


def hypervolume_2d(front_objs, ref_point):
    hv = 0.0
    prev_f2 = ref_point[1]
    for f1, f2 in sorted(front_objs):
        hv += (ref_point[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv


def simulated_annealing_multi(n_jobs, p, d, max_iter, p_accept=0.1):
    pi = list(range(n_jobs)); random.shuffle(pi)
    P, O = [], []  # wszystkie permutacje i ich cele
    for it in range(max_iter):
        i, j = random.sample(range(n_jobs), 2)
        pi_new = pi.copy(); pi_new[i], pi_new[j] = pi_new[j], pi_new[i]
        C_new = compute_completion_times(pi_new, p)
        o_new = objectives_two(C_new, d, pi_new)
        C_cur = compute_completion_times(pi, p)
        o_cur = objectives_two(C_cur, d, pi)
        if dominates(o_new, o_cur) or random.random() < p_accept:
            pi, o_cur = pi_new, o_new
        P.append(o_cur)
        O.append(o_cur)
    # front Pareto
    f_idx = pareto_front(O)
    front = [O[i] for i in f_idx]
    return O, front


def task1_experiments(n_jobs, seeds, max_iters):
    results = {}
    # najpierw globalny punkt referencyjny
    all_objs = []
    for max_iter in max_iters:
        for seed in seeds:
            p, d = generate_instance(n_jobs, seed)
            O, _ = simulated_annealing_multi(n_jobs, p, d, max_iter)
            all_objs.extend(O)
    z1 = max(o[0] for o in all_objs)
    z2 = max(o[1] for o in all_objs)
    ref_point = (z1, z2)
    # uruchomienia właściwe
    for max_iter in max_iters:
        fronts = []
        H = []
        for seed in seeds:
            p, d = generate_instance(n_jobs, seed)
            O, front = simulated_annealing_multi(n_jobs, p, d, max_iter)
            hv = hypervolume_2d(front, ref_point)
            fronts.append(front)
            H.append(hv)
        results[max_iter] = {'fronts': fronts, 'mean_hv': np.mean(H)}
    return results


def plot_subplots(results, max_iters):
    n = len(max_iters)
    cols = 3; rows = math.ceil((n+1)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()
    # osobne fronty dla każdego max_iter
    for idx, max_iter in enumerate(max_iters):
        ax = axes[idx]
        for front in results[max_iter]['fronts']:
            x, y = zip(*front) if front else ([],[])
            ax.scatter(x, y, color='red', s=10)
        ax.set_title(f'Front Pareto (it={max_iter})')
        ax.set_xlabel('Tmax'); ax.set_ylabel('ΣT')
        ax.grid(True)
    # wykres zbiorczy wszystkich frontów
    ax = axes[len(max_iters)]
    for max_iter in max_iters:
        for front in results[max_iter]['fronts']:
            x, y = zip(*front) if front else ([],[])
            ax.scatter(x, y, label=f'{max_iter} it', s=10)
    ax.set_title('Porównanie frontów')
    ax.set_xlabel('Tmax'); ax.set_ylabel('ΣT')
    ax.legend(); ax.grid(True)
    # wyłącz pusty subplot
    for j in range(len(max_iters)+1, rows*cols):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

# ---------------------------- Główna część ----------------------------
if __name__ == "__main__":
    n_jobs = 20
    seeds = list(range(10))
    max_iters = [100, 200, 400, 800, 1600]
    res1 = task1_experiments(n_jobs, seeds, max_iters)
    print("Task1 mean HV:", {k: res1[k]['mean_hv'] for k in max_iters})
    plot_subplots(res1, max_iters)
