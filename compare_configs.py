from solver import BeeTSP
import numpy as np

def run_config(name, bees, sites, limit):
    print(f"Running {name}: Bees={bees}, Sites={sites}, Limit={limit}")
    costs = []
    for _ in range(3):
        solver = BeeTSP(bees, sites, limit)
        _, history = solver.run()
        costs.append(history[-1])
    avg = np.mean(costs)
    print(f"  Avg Cost: {avg:.2f}")
    return avg

if __name__ == "__main__":
    cost_user = run_config("User Config", 100, 25, 1000)
    cost_bot = run_config("Bot Config", 150, 40, 2000)
    print(f"\nImprovement: {cost_user - cost_bot:.2f} ({(cost_user - cost_bot)/cost_user*100:.1f}%)")
