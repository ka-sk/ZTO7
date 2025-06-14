import random
import math
import numpy as np
import matplotlib.pyplot as plt
from chernoff import draw_faces 

# =============================================================================
# Laboratorium 7: Ocena i wizualizacja rozwiązań wielokryterialnych
# Zadania:
#  1) SA + Pareto front + HVI dla kryteriów 3 i 4
#  2) Skalaryzacja + SA dla kryteriów 3,4,5
#  3) Wizualizacje (metody 1,4,6,7) dla kryteriów 3,4,5,6
#
# Kryteria (minimalizacja):
#  1. Cmax (makespan)
#  2. ΣF (total flowtime)
#  3. Tmax (max tardiness)
#  4. ΣT (total tardiness)
#  5. Lmax (max lateness)
#  6. ΣL (total lateness)
# =============================================================================

# ---------------------------- Generacja instancji ----------------------------
def generate_instance(n_jobs, seed=None):
    """
    Generuje czasy przetwarzania p[i][j] i terminy d[j] zgodnie z opisem.
    """
    if seed is not None:
        random.seed(seed)
    # maszyny m=3
    p = [[random.randint(1, 99) for _ in range(n_jobs)] for _ in range(3)]
    S = sum(sum(row) for row in p)
    d = [random.randint(S//4, S//2) for _ in range(n_jobs)]
    return p, d

# ---------------------------- Obliczanie czasów zakończenia ----------------------------
def compute_completion_times(pi, p):
    m = len(p)
    n = len(pi)
    C = np.zeros((m, n), dtype=int)
    C[0, 0] = p[0][pi[0]]
    for j in range(1, n):
        C[0, j] = C[0, j-1] + p[0][pi[j]]
    for i in range(1, m):
        C[i, 0] = C[i-1, 0] + p[i][pi[0]]
        for j in range(1, n):
            C[i, j] = max(C[i, j-1], C[i-1, j]) + p[i][pi[j]]
    return C

# ---------------------------- Funkcje celu ----------------------------
def objectives_two(C, d, pi):
    """Kryteria 3 (Tmax) i 4 (ΣT)."""
    tardiness = [max(int(C[-1, j] - d[pi[j]]), 0) for j in range(len(pi))]
    Tmax = max(tardiness)
    SigmaT = sum(tardiness)
    return (Tmax, SigmaT)

def objectives_three(C, d, pi):
    """Kryteria 3 (Tmax), 4 (ΣT), 5 (Lmax)."""
    lateness = [int(C[-1, j] - d[pi[j]]) for j in range(len(pi))]
    tardiness = [max(l, 0) for l in lateness]
    Tmax = max(tardiness)
    SigmaT = sum(tardiness)
    Lmax = max(lateness)
    return (Tmax, SigmaT, Lmax)

def objectives_four(C, d, pi):
    """Kryteria 3 (Tmax), 4 (ΣT), 5 (Lmax), 6 (ΣL)."""
    lateness = [int(C[-1, j] - d[pi[j]]) for j in range(len(pi))]
    tardiness = [max(l, 0) for l in lateness]
    Tmax = max(tardiness)
    SigmaT = sum(tardiness)
    Lmax = max(lateness)
    SigmaL = sum(lateness)
    return (Tmax, SigmaT, Lmax, SigmaL)

# ---------------------------- Dominacja i front Pareto ----------------------------
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def pareto_front(pop, objs):
    front = []
    for i, o_i in enumerate(objs):
        dominated = False
        for j, o_j in enumerate(objs):
            if j != i and dominates(o_j, o_i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front

# ---------------------------- Hypervolume (2D) ----------------------------
def hypervolume_2d(front_objs, ref_point):
    hv = 0.0
    prev_f2 = ref_point[1]
    for f1, f2 in sorted(front_objs):
        hv += (ref_point[0] - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv

# ---------------------------- SA wielokryterialne (zad.1) ----------------------------
def simulated_annealing_multi(n_jobs, p, d, max_iter, p_accept=0.1):
    random.seed()
    pi = list(range(n_jobs)); random.shuffle(pi)
    P, O = [], []
    for it in range(max_iter):
        i, j = random.sample(range(n_jobs), 2)
        pi_new = pi.copy(); pi_new[i], pi_new[j] = pi_new[j], pi_new[i]
        C_new = compute_completion_times(pi_new, p)
        o_new = objectives_two(C_new, d, pi_new)
        C_cur = compute_completion_times(pi, p)
        o_cur = objectives_two(C_cur, d, pi)
        if dominates(o_new, o_cur) or random.random() < p_accept:
            pi = pi_new; o_cur = o_new
        P.append(pi.copy()); O.append(o_cur)
    idx = pareto_front(P, O)
    return P, O, [P[i] for i in idx], [O[i] for i in idx]

# ---------------------------- Eksperymenty zad.1 ----------------------------
def task1_experiments(n_jobs, seeds, max_iters):
    results = {}
    for max_iter in max_iters:
        all_fronts, all_hv = [], []
        # zbierz globalny ref point
        all_objs = []
        for seed in seeds:
            p, d = generate_instance(n_jobs, seed)
            _, O, _, OF = simulated_annealing_multi(n_jobs, p, d, max_iter)
            all_objs += O
        z1 = max(o[0] for o in all_objs)*1.0
        z2 = max(o[1] for o in all_objs)*1.0
        for seed in seeds:
            p, d = generate_instance(n_jobs, seed)
            _, _, _, OF = simulated_annealing_multi(n_jobs, p, d, max_iter)
            hv = hypervolume_2d(OF, (z1, z2))
            all_hv.append(hv); all_fronts.append(OF)
        results[max_iter] = {'mean_hv': np.mean(all_hv), 'fronts': all_fronts}
    return results

# ---------------------------- SA skalaryzowane (zad.2) ----------------------------
def simulated_annealing_scalar(n_jobs, p, d, max_iter, weights, p_accept=0.1):
    random.seed()
    pi = list(range(n_jobs)); random.shuffle(pi)
    xbest = pi.copy(); sbest = float('inf')
    def s_val(pi):
        C = compute_completion_times(pi, p)
        o = objectives_three(C, d, pi)
        return sum(w*v for w,v in zip(weights, o))
    for it in range(max_iter):
        i, j = random.sample(range(n_jobs), 2)
        pi_new = pi.copy(); pi_new[i], pi_new[j] = pi_new[j], pi_new[i]
        s_new = s_val(pi_new)
        if s_new < s_val(pi) or random.random() < p_accept:
            pi = pi_new
        if s_new < sbest:
            xbest, sbest = pi_new.copy(), s_new
    return xbest, sbest

# ---------------------------- Wizualizacje (zad.3) ----------------------------
def visualize_bar(solutions, criteria_vals):
    n= len(solutions); k= len(criteria_vals[0])
    fig, ax = plt.subplots()
    indices = np.arange(k)
    width = 0.8/n
    for i, vals in enumerate(criteria_vals):
        ax.bar(indices + i*width, vals, width, label=f'Sol {i+1}')
    ax.set_xticks(indices + width*(n-1)/2)
    ax.set_xticklabels([f'Kryterium {c}' for c in [3,4,5,6]])
    ax.set_ylabel('Wartość znormalizowana')
    ax.legend(); plt.show()

def visualize_star(solutions, criteria_vals):
    # współrzędne gwiazdowe
    angles = np.linspace(0, 2*np.pi, len(criteria_vals[0]), endpoint=False).tolist()
    angles += angles[:1]
    fig, axes = plt.subplots(1, len(solutions), subplot_kw=dict(polar=True), figsize=(4*len(solutions),4))
    for ax, vals in zip(axes, criteria_vals):
        data = vals + vals[:1]
        ax.plot(angles, data, 'o-')
        ax.fill(angles, data, alpha=0.2)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels([f'{c}' for c in [3,4,5,6]])
        ax.set_ylim(0,1)
    plt.show()

def visualize_spider(solutions, criteria_vals):
    # wykresy pajęczynowe (radar chart)
    angles = np.linspace(0, 2*np.pi, len(criteria_vals[0]), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111, polar=True)
    for vals in criteria_vals:
        data = vals + vals[:1]
        ax.plot(angles, data, linewidth=1)
        ax.fill(angles, data, alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([f'{c}' for c in [3,4,5,6]])
    ax.set_ylim(0,1); plt.show()

# Zadanie główne: zbierz 3 z frontu i 1 słabsze oraz uruchom:
def task3_visualization(n_jobs, seed=0):
    p, d = generate_instance(n_jobs, seed)
    # uzyskaj front czterokryterialny
    P, O, F, OF = simulated_annealing_multi(n_jobs, p, d, 1000)
    # weź 3 pierwsze z F lub uzupełnij jeśli mniej
    sels = F[:3]
    if len(sels) < 3: sels += [P[0]]*(3-len(sels))
    # słabsze: początkowe losowe
    weak = list(range(n_jobs)); random.shuffle(weak)
    sols = sels + [weak]
    # oblicz wartości czterech kryteriów
    vals = [objectives_four(compute_completion_times(pi,p), d, pi) for pi in sols]
    # normalizacja do [0,1]
    arr = np.array(vals, dtype=float)
    minv = arr.min(axis=0); maxv = arr.max(axis=0)
    norm = (arr - minv)/(maxv - minv)
    # wizualizacje:
    visualize_bar(sols, norm.tolist())
    visualize_star(sols, norm.tolist())
    visualize_spider(sols, norm.tolist())
    # twarze Chernoffa
    draw_faces(norm.tolist())

if __name__ == "__main__":
    n_jobs = 20
    seeds = list(range(10))
    max_iters = [100,200,400,800,1600]
    # Zad.1
    res1 = task1_experiments(n_jobs, seeds, max_iters)
    print("Task1 mean HV:", {k:res1[k]['mean_hv'] for k in max_iters})
    # Zad.2
    weights = [0.3,0.3,0.4]
    for mi in max_iters:
        p,d = generate_instance(n_jobs,0)
        xb,sb = simulated_annealing_scalar(n_jobs,p,d,mi,weights)
        print(f"max_iter={mi}, s_best={sb}")
    # Zad.3
    task3_visualization(n_jobs)
