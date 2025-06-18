from collections import deque
import random
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from RandomNumberGenerator import RandomNumberGenerator


def generate_instance(n, seed):
    """Generate a problem instance with n jobs and given seed"""
    rng = RandomNumberGenerator(seed)
    p = [[0 for _ in range(n)] for _ in range(3)]
    S = 0

    # Generate processing times
    for i in range(3):
        for j in range(n):
            p[i][j] = rng.nextInt(1, 99)
            S += p[i][j]

    # Generate due dates
    d = []
    for j in range(n):
        d.append(rng.nextInt(S // 4, S // 2))

    return p, d


def calculate_completion_times(pi, p):
    """Calculate completion times for all jobs in schedule pi"""
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
    """Calculate various criteria for a schedule pi"""
    C = calculate_completion_times(pi, p)
    n = len(pi)

    # Criterion 1: Maximum completion time (makespan)
    cmax = max(C[2])

    # Criterion 3: Maximum tardiness
    max_tardiness = float("-inf")
    for j in range(n):
        completion_time = C[2][j]
        job_index = pi[j]
        tardiness = max(completion_time - d[job_index], 0)
        max_tardiness = max(max_tardiness, tardiness)

    # Criterion 4: Total tardiness
    sum_tardiness = 0
    for j in range(n):
        completion_time = C[2][j]
        job_index = pi[j]
        tardiness = max(completion_time - d[job_index], 0)
        sum_tardiness += tardiness

    return cmax, max_tardiness, sum_tardiness


def get_neighbor(pi, method="swap"):
    """Generate a neighboring solution by swapping or inserting jobs"""
    n = len(pi)
    new_pi = pi.copy()

    if method == "swap" or (method == "mixed" and random.random() < 0.5):
        # Swap operation
        i, j = random.sample(range(n), 2)
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]
    else:
        # Insert operation
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            if i > j:
                i, j = j, i
            element = new_pi.pop(i)
            new_pi.insert(j, element)

    return new_pi


def simulated_annealing(
    p, d, criterion_idx, max_iter=10000, T0=100, alpha=0.99, neighborhood_method="mixed"
):
    n = len(p[0])

    # Initialize with a random permutation
    pi = list(range(n))
    random.shuffle(pi)

    # Calculate initial criteria
    cmax, max_tardiness, sum_tardiness = calculate_criteria(pi, p, d)

    # Select the criterion to optimize
    if criterion_idx == 1:  # Max Tardiness
        current_criterion = max_tardiness
    else:  # Total Tardiness
        current_criterion = sum_tardiness

    # Initialize best solution
    best_pi = pi.copy()
    best_criterion = current_criterion
    current_pi = pi.copy()

    # Initialize temperature and history
    T = T0
    history = [current_criterion]

    # Main loop
    for iter in range(max_iter):
        # Generate neighbor
        neighbor_pi = get_neighbor(current_pi, method=neighborhood_method)

        # Calculate criteria for neighbor
        _, neighbor_max_tardiness, neighbor_sum_tardiness = calculate_criteria(
            neighbor_pi, p, d
        )

        # Select criterion to compare
        if criterion_idx == 1:  # Max Tardiness
            neighbor_criterion = neighbor_max_tardiness
        else:  # Total Tardiness
            neighbor_criterion = neighbor_sum_tardiness

        # Calculate delta
        delta = neighbor_criterion - current_criterion

        # Decide whether to accept the neighbor
        if delta <= 0 or random.random() < np.exp(-delta / T):
            current_pi = neighbor_pi
            current_criterion = neighbor_criterion

            # Update best solution if improved
            if current_criterion < best_criterion:
                best_pi = current_pi.copy()
                best_criterion = current_criterion

        # Cool down
        T *= alpha

        # Save history
        history.append(best_criterion)

        # Early stopping if temperature is too low
        if T < 0.01:
            break

    return best_pi, best_criterion, history


def print_schedule_details(pi, p, d, criterion_name):
    """Print detailed information about a schedule"""
    n = len(pi)
    C = calculate_completion_times(pi, p)

    print(f"\n{criterion_name} Schedule Details:")
    print("-" * 80)
    print(
        f"{'Job ID':^10} | {'Machine 1':^10} | {'Machine 2':^10} | {'Machine 3':^10} | {'Due Date':^10} | {'Completion':^10} | {'Tardiness':^10}"
    )
    print("-" * 80)

    total_tardiness = 0
    max_tardiness = float("-inf")

    for j in range(n):
        job_idx = pi[j]
        completion_time = C[2][j]
        tardiness = max(completion_time - d[job_idx], 0)

        total_tardiness += tardiness
        max_tardiness = max(max_tardiness, tardiness)

        print(
            f"{job_idx:^10} | {p[0][job_idx]:^10} | {p[1][job_idx]:^10} | {p[2][job_idx]:^10} | {d[job_idx]:^10} | {completion_time:^10} | {tardiness:^10}"
        )

    print("-" * 80)
    print(f"Maximum Tardiness: {max_tardiness}")
    print(f"Total Tardiness: {total_tardiness}")
    print("-" * 80)


def run_experiments(n, seed):
    """Run experiments for both criteria"""
    p, d = generate_instance(n, seed)

    # Parameters
    max_iter = 10000
    T0 = 100
    alpha = 0.99
    neighborhood_method = "mixed"

    # Solve for max tardiness (criterion 3)
    start_time = time.time()
    best_pi_max, best_max_tardiness, history_max = simulated_annealing(
        p,
        d,
        criterion_idx=1,
        max_iter=max_iter,
        T0=T0,
        alpha=alpha,
        neighborhood_method=neighborhood_method,
    )
    max_tardiness_time = time.time() - start_time

    # Solve for total tardiness (criterion 4)
    start_time = time.time()
    best_pi_sum, best_sum_tardiness, history_sum = simulated_annealing(
        p,
        d,
        criterion_idx=2,
        max_iter=max_iter,
        T0=T0,
        alpha=alpha,
        neighborhood_method=neighborhood_method,
    )
    sum_tardiness_time = time.time() - start_time

    # Print results
    print(f"\nResults for n={n}, seed={seed}:")
    print(f"Max Tardiness: {best_max_tardiness} (Time: {max_tardiness_time:.2f}s)")
    print(f"Total Tardiness: {best_sum_tardiness} (Time: {sum_tardiness_time:.2f}s)")

    # Print detailed schedules
    print_schedule_details(best_pi_max, p, d, "Max Tardiness")
    print_schedule_details(best_pi_sum, p, d, "Total Tardiness")

    # Plot convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_max)
    plt.title(f"Max Tardiness Convergence (n={n})")
    plt.xlabel("Iteration")
    plt.ylabel("Max Tardiness")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_sum)
    plt.title(f"Total Tardiness Convergence (n={n})")
    plt.xlabel("Iteration")
    plt.ylabel("Total Tardiness")
    plt.grid(True)

    # Create results directory if it doesn't exist
    os.makedirs(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        exist_ok=True,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results",
            f"task1_convergence_n{n}_s{seed}.png",
        )
    )
    plt.close()

    return best_pi_max, best_max_tardiness, best_pi_sum, best_sum_tardiness


def is_dominated(solution, solutions):
    max_tardiness, sum_tardiness = solution

    for other_max, other_sum in solutions:
        # Check if other solution dominates this one (better in at least one objective, not worse in any)
        if (other_max <= max_tardiness and other_sum < sum_tardiness) or (
            other_max < max_tardiness and other_sum <= sum_tardiness
        ):
            return True

    return False


def find_pareto_front(solutions):
    pareto_front = []

    # Check each solution
    for solution in solutions:
        # Extract criteria values (excluding the solution itself)
        criteria = (solution[0], solution[1])

        # Extract other solutions' criteria values
        other_solutions = [(s[0], s[1]) for s in pareto_front]

        # If the solution is not dominated by any other solution, add it to the Pareto front
        if not is_dominated(criteria, other_solutions):
            # Remove any solutions in the Pareto front that are dominated by this solution
            new_pareto_front = []
            for idx, p_solution in enumerate(pareto_front):
                if not is_dominated((p_solution[0], p_solution[1]), [criteria]):
                    new_pareto_front.append(p_solution)

            pareto_front = new_pareto_front
            pareto_front.append(solution)

    return pareto_front


def generate_random_solutions(p, d, n_solutions=1000):
    solutions = []
    n = len(p[0])

    for _ in range(n_solutions):
        pi = list(range(n))
        random.shuffle(pi)

        _, max_tardiness, sum_tardiness = calculate_criteria(pi, p, d)
        solutions.append((max_tardiness, sum_tardiness, pi))

    return solutions


def generate_solutions_with_sa(p, d, n_weights=20, max_iter=5000):
    """
    Generate solutions using weighted sum approach with simulated annealing

    Parameters:
    - p: processing times matrix
    - d: due dates list
    - n_weights: number of weight combinations to try
    - max_iter: maximum iterations for each SA run

    Returns:
    - solutions: list of tuples [(max_tardiness, sum_tardiness, pi), ...]
    """
    solutions = []
    n = len(p[0])

    # Generate a range of weights
    weights = np.linspace(0, 1, n_weights)

    for w1 in weights:
        w2 = 1 - w1

        # Define a custom simulated annealing function for weighted sum
        def weighted_simulated_annealing(
            p, d, w1, w2, max_iter=5000, T0=100, alpha=0.99
        ):
            n = len(p[0])

            # Initialize with a random permutation
            pi = list(range(n))
            random.shuffle(pi)

            # Calculate initial criteria
            _, max_tardiness, sum_tardiness = calculate_criteria(pi, p, d)

            # Weighted sum
            current_criterion = w1 * max_tardiness + w2 * sum_tardiness

            # Initialize best solution
            best_pi = pi.copy()
            best_criterion = current_criterion
            current_pi = pi.copy()

            # Initialize temperature
            T = T0

            # Main loop
            for iter in range(max_iter):
                # Generate neighbor
                neighbor_pi = get_neighbor(current_pi, method="mixed")

                # Calculate criteria for neighbor
                _, neighbor_max_tardiness, neighbor_sum_tardiness = calculate_criteria(
                    neighbor_pi, p, d
                )

                # Weighted sum
                neighbor_criterion = (
                    w1 * neighbor_max_tardiness + w2 * neighbor_sum_tardiness
                )

                # Calculate delta
                delta = neighbor_criterion - current_criterion

                # Decide whether to accept the neighbor
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_pi = neighbor_pi
                    current_criterion = neighbor_criterion

                    # Update best solution if improved
                    if current_criterion < best_criterion:
                        best_pi = current_pi.copy()
                        best_criterion = current_criterion

                # Cool down
                T *= alpha

                # Early stopping if temperature is too low
                if T < 0.01:
                    break

            # Calculate the actual criteria values for the best solution
            _, best_max_tardiness, best_sum_tardiness = calculate_criteria(
                best_pi, p, d
            )
            return best_pi, best_max_tardiness, best_sum_tardiness

        # Run simulated annealing with current weights
        best_pi, max_tardiness, sum_tardiness = weighted_simulated_annealing(
            p, d, w1, w2, max_iter
        )
        solutions.append((max_tardiness, sum_tardiness, best_pi))

    return solutions


def plot_pareto_front(
    pareto_front, random_solutions=None, sa_solutions=None, save_path=None
):
    plt.figure(figsize=(10, 8))

    # Extract criteria values
    pareto_max = [solution[0] for solution in pareto_front]
    pareto_sum = [solution[1] for solution in pareto_front]

    # Sort points for connecting line
    pareto_points = list(zip(pareto_max, pareto_sum))
    pareto_points.sort()
    pareto_max, pareto_sum = zip(*pareto_points)

    # Plot Pareto front
    plt.plot(
        pareto_max, pareto_sum, "ro-", linewidth=2, markersize=10, label="Pareto Front"
    )

    # Plot random solutions if provided
    if random_solutions:
        random_max = [solution[0] for solution in random_solutions]
        random_sum = [solution[1] for solution in random_solutions]
        plt.scatter(
            random_max, random_sum, color="blue", alpha=0.3, label="Random Solutions"
        )

    # Plot SA solutions if provided
    if sa_solutions:
        sa_max = [solution[0] for solution in sa_solutions]
        sa_sum = [solution[1] for solution in sa_solutions]
        plt.scatter(sa_max, sa_sum, color="green", alpha=0.6, label="SA Solutions")

    # Customize plot
    plt.title("Pareto Front: Max Tardiness vs Total Tardiness", fontsize=16)
    plt.xlabel("Max Tardiness", fontsize=14)
    plt.ylabel("Total Tardiness", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    # Add annotations for Pareto optimal solutions
    for i, (max_t, sum_t) in enumerate(zip(pareto_max, pareto_sum)):
        plt.annotate(
            f"P{i+1}",
            (max_t, sum_t),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def run_pareto_analysis(n, seed):
    """
    Run Pareto front analysis for the flow shop scheduling problem

    Parameters:
    - n: number of jobs
    - seed: random seed for instance generation

    Returns:
    - pareto_front: list of Pareto optimal solutions
    """
    # Generate instance
    p, d = generate_instance(n, seed)

    print(f"\nRunning Pareto front analysis for n={n}, seed={seed}...")

    # Generate random solutions
    print("Generating random solutions...")
    random_solutions = generate_random_solutions(p, d, n_solutions=500)

    # Generate solutions with simulated annealing
    print("Generating solutions with simulated annealing...")
    sa_solutions = generate_solutions_with_sa(p, d, n_weights=20, max_iter=5000)

    # Combine all solutions
    all_solutions = random_solutions + sa_solutions

    # Find Pareto front
    print("Finding Pareto front...")
    pareto_front = find_pareto_front(all_solutions)

    # Sort Pareto front by max tardiness
    pareto_front.sort(key=lambda x: x[0])

    # Print Pareto optimal solutions
    print(f"\nFound {len(pareto_front)} Pareto optimal solutions:")
    print("-" * 80)
    print(f"{'Solution':^10} | {'Max Tardiness':^15} | {'Total Tardiness':^15}")
    print("-" * 80)

    for i, solution in enumerate(pareto_front):
        max_tardiness, sum_tardiness, _ = solution
        print(f"P{i+1:^9} | {max_tardiness:^15} | {sum_tardiness:^15}")

    print("-" * 80)

    # Create results directory if it doesn't exist
    os.makedirs(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        exist_ok=True,
    )

    # Plot Pareto front
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"pareto_front_n{n}_s{seed}.png",
    )
    plot_pareto_front(pareto_front, random_solutions, sa_solutions, save_path)
    print(f"Pareto front plot saved to {save_path}")

    # Export detailed solutions to a text file
    text_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"pareto_solutions_n{n}_s{seed}.txt",
    )
    with open(text_path, "w") as f:
        f.write(f"Pareto optimal solutions for n={n}, seed={seed}\n")
        f.write("-" * 80 + "\n")

        for i, solution in enumerate(pareto_front):
            max_tardiness, sum_tardiness, pi = solution
            f.write(f"Solution P{i+1}:\n")
            f.write(f"Max Tardiness: {max_tardiness}\n")
            f.write(f"Total Tardiness: {sum_tardiness}\n")
            f.write(f"Schedule: {pi}\n")
            f.write("-" * 80 + "\n")

    print(f"Detailed solutions saved to {text_path}")

    return pareto_front


def calculate_hypervolume_2d(pareto_front, nadir_point):
    """
    Calculate hypervolume for a 2D Pareto front

    Parameters:
    - pareto_front: list of tuples [(max_tardiness, sum_tardiness, pi), ...]
    - nadir_point: reference point (z1, z2)

    Returns:
    - Hypervolume value
    """
    if not pareto_front:
        return 0.0

    # Extract only the criteria values and sort by the first criterion
    objectives = [(solution[0], solution[1]) for solution in pareto_front]
    objectives.sort(key=lambda x: x[0])

    # Calculate hypervolume using rectangle algorithm
    hypervolume = 0.0

    # Start from point (0, nadir[1])
    y_right = nadir_point[1]

    for i, (x, y) in enumerate(objectives):
        # Add rectangle from previous x to current x
        width = nadir_point[0] - x
        height = y_right - y

        if width > 0 and height > 0:
            hypervolume += width * height

        # Update previous y
        y_right = y

    return hypervolume


def calculate_hvi(fronts, nadir_factor=1.2):
    """
    Calculate Hyper Volume Indicator (HVI) for a list of Pareto fronts

    Parameters:
    - fronts: list of Pareto fronts, where each front is a list of solutions
    - nadir_factor: factor to calculate the reference point (nadir)

    Returns:
    - List of HVI values for each front
    """
    if not fronts:
        return []

    # Find reference point (nadir) - worst values from all fronts
    all_objectives = []
    for front in fronts:
        for solution in front:
            all_objectives.append((solution[0], solution[1]))

    if not all_objectives:
        return [0.0] * len(fronts)

    # Find maximum values for both criteria
    max_tardiness = max(obj[0] for obj in all_objectives) if all_objectives else 0
    max_sum_tardiness = max(obj[1] for obj in all_objectives) if all_objectives else 0

    # Reference point (nadir)
    nadir_point = (max_tardiness * nadir_factor, max_sum_tardiness * nadir_factor)

    # Calculate HVI for each front
    hvi_values = []
    for i, front in enumerate(fronts):
        if not front:
            hvi_values.append(0.0)
            continue

        hvi = calculate_hypervolume_2d(front, nadir_point)
        hvi_values.append(hvi)

    return hvi_values


def run_hvi_experiments(n, seed, max_iter_values=None, num_runs=10):
    """
    Run experiments with different max_iter values and calculate HVI

    Parameters:
    - n: number of jobs
    - seed: random seed for instance generation
    - max_iter_values: list of max_iter values to test
    - num_runs: number of runs for each max_iter value

    Returns:
    - Dictionary with HVI values for each max_iter
    """
    if max_iter_values is None:
        max_iter_values = [100, 200, 400, 800, 1600]

    # Generate seeds for different runs
    run_seeds = [seed + i * 10 for i in range(num_runs)]

    results = {}

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    # File to save HVI values
    hvi_file_path = os.path.join(results_dir, "hvi_multiple_runs.txt")

    with open(hvi_file_path, "w") as f:
        f.write("Wartości HVI dla różnych wartości maxIter i 10 przebiegów\n")
        f.write("-" * 80 + "\n\n")
        f.write(
            "Format: maxIter, [HVI dla przebiegu 1, HVI dla przebiegu 2, ...], średnia HVI\n\n"
        )

    for max_iter in max_iter_values:
        print(f"\nRunning experiments with max_iter={max_iter}...")
        fronts = []

        for i, run_seed in enumerate(run_seeds):
            print(f"  Run {i+1}/{num_runs} with seed {run_seed}")

            # Generate instance
            p, d = generate_instance(n, run_seed)

            # Generate solutions with SA using weighted sum approach
            sa_solutions = generate_solutions_with_sa(
                p, d, n_weights=10, max_iter=max_iter
            )

            # Find Pareto front
            pareto_front = find_pareto_front(sa_solutions)
            fronts.append(pareto_front)

            # Plot and save Pareto front for this run
            save_path = os.path.join(
                results_dir, f"pareto_front_n{n}_maxiter{max_iter}_run{i+1}.png"
            )
            plot_pareto_front(
                pareto_front, sa_solutions=sa_solutions, save_path=save_path
            )

        # Calculate HVI for all fronts
        hvi_values = calculate_hvi(fronts)
        avg_hvi = np.mean(hvi_values)
        results[max_iter] = {"hvi_values": hvi_values, "avg_hvi": avg_hvi}

        # Save HVI values to file
        with open(hvi_file_path, "a") as f:
            f.write(f"maxIter = {max_iter}:\n")
            f.write(f"  Wartości HVI: {hvi_values}\n")
            f.write(f"  Średnia HVI: {avg_hvi:.6f}\n")
            f.write("-" * 80 + "\n\n")

        print(f"  Average HVI: {avg_hvi}")

    # Plot average HVI values
    plt.figure(figsize=(10, 6))
    avg_hvi_values = [results[max_iter]["avg_hvi"] for max_iter in max_iter_values]
    plt.plot(max_iter_values, avg_hvi_values, "o-", linewidth=2, markersize=8)
    plt.title("Average HVI vs. max_iter", fontsize=16)
    plt.xlabel("max_iter", fontsize=14)
    plt.ylabel("Average HVI", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "hvi_vs_maxiter.png"), dpi=300)
    plt.close()

    print(f"\nResults saved to {hvi_file_path}")
    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Problem parameters
    n = 100  # Number of jobs
    seed = 1  # Seed for instance generation

    # Choose what to run
    run_single_objective = False
    run_pareto = False
    run_hvi = True

    if run_single_objective:
        # Run single-objective experiments
        print("Running single-objective optimization...")
        best_pi_max, best_max_tardiness, best_pi_sum, best_sum_tardiness = (
            run_experiments(n, seed)
        )
        print("\nSingle-objective optimization completed!")

    if run_pareto:
        # Run Pareto front analysis
        print("\nRunning Pareto front analysis...")
        pareto_front = run_pareto_analysis(n, seed)
        print("\nPareto front analysis completed!")

    if run_hvi:
        # Run HVI experiments with different max_iter values
        print("\nRunning HVI experiments...")
        max_iter_values = [100, 200, 400, 800, 1600]
        hvi_results = run_hvi_experiments(n, seed, max_iter_values, num_runs=10)
        print("\nHVI experiments completed!")

    print("\nAll experiments completed successfully!")
