from multiprocessing import Pool, cpu_count
import time

def f(x):
    time.sleep(1)
    return x * x

if __name__ == "__main__":  # REQUIRED on macOS & Windows
    with Pool(cpu_count()) as pool:
        results = pool.map(f, range(10))

    print(results)


def simulate(a, b):
    time.sleep(2)
    return a * b

if __name__ == "__main__":
    params = [(1, 2), (3, 4), (5, 6)]
    with Pool(cpu_count()) as pool:
        results = pool.starmap(simulate, params)
    print(results)  # [2, 12, 30]

from multiprocessing import Pool, cpu_count
from itertools import product

# -------------------------
# Worker function
# -------------------------
def compute(n, q, x, params):
    """Compute a simple expression using both per-job and global parameters."""
    result = n + x * q + params["offset"]
    return (n, q, result)

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":

    # Fixed "global" parameters
    x = 5.0
    global_params = {
        "offset": 10,
        "experiment": "2D_grid_demo"
    }

    # Parameter sweep ranges
    n_values = [0, 1, 2, 3, 4]
    q_values = [0.1, 0.5, 1.0, 2.0]

    # Generate all combinations (Cartesian product)
    param_grid = list(product(n_values, q_values))

    # Add shared parameters to each tuple
    # Each element becomes (n, q, x, params)
    args = [(n, q, x, global_params) for (n, q) in param_grid]

    # Parallel computation
    with Pool(cpu_count()) as pool:
        results = pool.starmap(compute, args)

    # Show results
    print("Results (n, q, result):")
    for r in results:
        print(r)