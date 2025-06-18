import os
import random

import matplotlib.pyplot as plt
import numpy as np

from RandomNumberGenerator import RandomNumberGenerator


def generate_instance(n, seed):
    rng = RandomNumberGenerator(seed)
    p = [[0 for _ in range(n)] for _ in range(3)]
    S = 0

    for i in range(3):
        for j in range(n):
            p[i][j] = rng.nextInt(1, 99)
            S += p[i][j]

    d = []
    for j in range(n):
        d.append(rng.nextInt(S // 4, S // 2))

    return p, d


def calculate_completion_times(pi, p):
    n = len(pi)
    C = [[0 for _ in range(n)] for _ in range(3)]

    for j in range(n):
        for i in range(3):
            if j == 0 and i == 0:
                C[i][j] = p[i][pi[j]]
            elif j == 0 and i > 0:
                C[i][j] = C[i - 1][j] + p[i][pi[j]]
            elif j > 0 and i == 0:
                C[i][j] = C[i][j - 1] + p[i][pi[j]]
            else:
                C[i][j] = max(C[i - 1][j], C[i][j - 1]) + p[i][pi[j]]

    return C


def calculate_criteria(pi, p, d):
    C = calculate_completion_times(pi, p)
    n = len(pi)

    # Criterion 3: Maximum tardiness
    max_tardiness = float("-inf")
    for j in range(n):
        completion_time = C[2][j]
        job_index = pi[j]
        tardiness = max(completion_time - d[job_index], 0)
        max_tardiness = max(max_tardiness, tardiness)

    # Criterion 4: Total tardiness
    total_tardiness = 0
    for j in range(n):
        completion_time = C[2][j]
        job_index = pi[j]
        tardiness = max(completion_time - d[job_index], 0)
        total_tardiness += tardiness

    # Criterion 5: Maximum lateness
    max_lateness = float("-inf")
    for j in range(n):
        completion_time = C[2][j]
        job_index = pi[j]
        lateness = completion_time - d[job_index]
        max_lateness = max(max_lateness, lateness)

    return max_tardiness, total_tardiness, max_lateness


def get_neighbor(pi):
    n = len(pi)
    new_pi = pi.copy()

    if random.random() < 0.5:
        i, j = random.sample(range(n), 2)
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    else:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            if i > j:
                i, j = j, i
            element = new_pi.pop(i)
            new_pi.insert(j, element)

    return new_pi


def scalarize(max_tardiness, total_tardiness, max_lateness, c1, c2, c3):
    return c1 * max_tardiness + c2 * total_tardiness + c3 * max_lateness


def simulated_annealing_scalar(p, d, max_iter, c1, c2, c3):
    n = len(p[0])
    pi = list(range(n))
    random.shuffle(pi)

    best_pi = pi.copy()
    max_tardiness, total_tardiness, max_lateness = calculate_criteria(pi, p, d)
    best_score = scalarize(max_tardiness, total_tardiness, max_lateness, c1, c2, c3)

    current_pi = pi.copy()
    current_score = best_score

    for it in range(max_iter):
        neighbor = get_neighbor(current_pi)
        max_tardiness_n, total_tardiness_n, max_lateness_n = calculate_criteria(neighbor, p, d)
        neighbor_score = scalarize(max_tardiness_n, total_tardiness_n, max_lateness_n, c1, c2, c3)

        p_accept = 0.995**it

        if neighbor_score < current_score:
            current_pi = neighbor
            current_score = neighbor_score
        elif random.random() < p_accept:
            current_pi = neighbor
            current_score = neighbor_score

        if current_score < best_score:
            best_pi = current_pi.copy()
            best_score = current_score

    return best_pi, best_score


def normalize_coefficients(sample_solutions, p, d):
    max_tardiness_values = []
    total_tardiness_values = []
    max_lateness_values = []

    for pi in sample_solutions:
        max_tardiness, total_tardiness, max_lateness = calculate_criteria(pi, p, d)
        max_tardiness_values.append(max_tardiness)
        total_tardiness_values.append(total_tardiness)
        max_lateness_values.append(max_lateness)

    avg_max_tardiness = np.mean(max_tardiness_values)
    avg_total_tardiness = np.mean(total_tardiness_values)
    avg_max_lateness = np.mean(max_lateness_values)

    c1 = 1.0
    c2 = avg_max_tardiness / avg_total_tardiness if avg_total_tardiness > 0 else 1.0
    c3 = avg_max_tardiness / avg_max_lateness if avg_max_lateness > 0 else 1.0

    return c1, c2, c3


def main():
    os.makedirs("results", exist_ok=True)

    n = 10
    seed = 42
    p, d = generate_instance(n, seed)

    sample_solutions = []
    for _ in range(10):
        pi = list(range(n))
        random.shuffle(pi)
        sample_solutions.append(pi)

    c1, c2, c3 = normalize_coefficients(sample_solutions, p, d)

    max_iter_values = [100, 200, 400, 800, 1600]
    results = {}

    for max_iter in max_iter_values:
        scores = []
        for run in range(10):
            random.seed(run * 100 + max_iter)
            best_pi, best_score = simulated_annealing_scalar(p, d, max_iter, c1, c2, c3)
            scores.append(best_score)

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        results[max_iter] = {
            "avg_score": avg_score,
            "std_score": std_score,
            "scores": scores,
        }

        print(f"maxIter={max_iter}: avg_score={avg_score:.2f}, std={std_score:.2f}")

    plt.figure(figsize=(10, 6))
    iterations = list(results.keys())
    avg_scores = [results[it]["avg_score"] for it in iterations]
    std_scores = [results[it]["std_score"] for it in iterations]

    plt.errorbar(iterations, avg_scores, yerr=std_scores, marker="o", capsize=5)
    plt.xlabel("Liczba iteracji (maxIter)")
    plt.ylabel("Średnia wartość funkcji skalaryzowanej")
    plt.title("Jakość rozwiązania w zależności od liczby iteracji")
    plt.grid(True)
    plt.savefig("results/task2_quality_vs_iterations.png", dpi=300, bbox_inches="tight")
    plt.close()

    coefficient_sets = [(1.0, 1.0, 1.0), (c1, c2, c3), (1.0, 0.5, 2.0), (2.0, 1.0, 0.5)]

    coeff_results = {}
    max_iter = 800

    for i, (c1_test, c2_test, c3_test) in enumerate(coefficient_sets):
        scores = []
        for run in range(10):
            random.seed(run * 100 + i * 1000)
            best_pi, best_score = simulated_annealing_scalar(
                p, d, max_iter, c1_test, c2_test, c3_test
            )
            scores.append(best_score)

        avg_score = np.mean(scores)
        coeff_results[f"c1={c1_test:.1f}, c2={c2_test:.1f}, c3={c3_test:.1f}"] = (
            avg_score
        )
        print(
            f"Coefficients c1={c1_test:.1f}, c2={c2_test:.1f}, c3={c3_test:.1f}: avg_score={avg_score:.2f}"
        )

    plt.figure(figsize=(12, 6))
    coeff_names = list(coeff_results.keys())
    coeff_scores = list(coeff_results.values())

    plt.bar(range(len(coeff_names)), coeff_scores)
    plt.xlabel("Zestawy współczynników")
    plt.ylabel("Średnia wartość funkcji skalaryzowanej")
    plt.title("Wpływ współczynników skalaryzacji na jakość rozwiązania")
    plt.xticks(range(len(coeff_names)), coeff_names, rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(
        "results/task2_coefficients_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    with open("results/task2_results.txt", "w") as f:
        f.write("Zadanie 2 - Wyniki algorytmu SA ze skalaryzacją\n")
        f.write("=" * 50 + "\n\n")

        f.write("Parametry problemu:\n")
        f.write(f"n = {n}\n")
        f.write(f"seed = {seed}\n")
        f.write(
            f"Znormalizowane współczynniki: c1={c1:.3f}, c2={c2:.3f}, c3={c3:.3f}\n\n"
        )
        f.write("Gdzie:\n")
        f.write("c1 - waga dla maksymalnego spóźnienia (max tardiness)\n")
        f.write("c2 - waga dla sumy spóźnień (total tardiness)\n")
        f.write("c3 - waga dla maksymalnego opóźnienia (max lateness)\n\n")

        f.write("Wyniki dla różnych wartości maxIter:\n")
        for max_iter in max_iter_values:
            f.write(f"maxIter={max_iter}: ")
            f.write(f"avg_score={results[max_iter]['avg_score']:.3f}, ")
            f.write(f"std={results[max_iter]['std_score']:.3f}\n")

        f.write("\nWyniki dla różnych zestawów współczynników:\n")
        for coeff_name, score in coeff_results.items():
            f.write(f"{coeff_name}: avg_score={score:.3f}\n")


if __name__ == "__main__":
    main()
