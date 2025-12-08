import numpy as np
import random
import time

NUM_CITIES = 300
np.random.seed(42) 
DIST_MATRIX = np.random.randint(5, 151, size=(NUM_CITIES, NUM_CITIES))
np.fill_diagonal(DIST_MATRIX, 0)

def calculate_cost(path):
    from_cities = path[:-1]
    to_cities = path[1:]
    dist = np.sum(DIST_MATRIX[from_cities, to_cities])
    dist += DIST_MATRIX[path[-1], path[0]]
    return dist

class BeeTSP:
    def __init__(self, num_bees, num_sites, limit_iterations):
        self.num_bees = num_bees
        self.num_sites = num_sites
        self.limit = limit_iterations
        self.best_solution = None
        self.best_cost = float('inf')
        self.history = []

    def run(self, progress_callback=None):
        population = []
        costs = []
        
        base_path = np.arange(NUM_CITIES)
        
        for _ in range(self.num_sites):
            p = np.random.permutation(base_path)
            c = calculate_cost(p)
            population.append(p)
            costs.append(c)
            if c < self.best_cost:
                self.best_cost = c
                self.best_solution = p

        for it in range(self.limit):
            for i in range(self.num_sites):
                p = population[i].copy()
                a, b = random.sample(range(NUM_CITIES), 2)
                p[a], p[b] = p[b], p[a]
                c = calculate_cost(p)
                if c < costs[i]:
                    population[i] = p
                    costs[i] = c
                    if c < self.best_cost:
                        self.best_cost = c
                        self.best_solution = p

            num_onlookers = self.num_bees - self.num_sites
            
            if num_onlookers > 0:
                sorted_idx = np.argsort(costs)
                
                best_sites_indices = sorted_idx[:max(1, self.num_sites // 2)]
                
                for _ in range(num_onlookers):
                    target_site_idx = random.choice(best_sites_indices)
                    
                    p = population[target_site_idx].copy()
                    a, b = random.sample(range(NUM_CITIES), 2)
                    p[a], p[b] = p[b], p[a]
                    c = calculate_cost(p)
                    
                    if c < costs[target_site_idx]:
                        population[target_site_idx] = p
                        costs[target_site_idx] = c
                        if c < self.best_cost:
                            self.best_cost = c
                            self.best_solution = p
            
            self.history.append(self.best_cost)
            
            if progress_callback and it % 10 == 0:
                progress_callback(it, self.best_cost)
        
        return self.best_solution, self.history

class ParameterTuner:
    def __init__(self):
        self.param_ranges = {
            'num_bees': range(10, 210, 20),
            'num_sites': range(5, 55, 5),
            'limit': range(500, 2500, 500)
        }
        self.current_params = {
            'num_bees': 100,
            'num_sites': 25,
            'limit': 1000
        }
        self.best_global_cost = float('inf')
        self.best_params = self.current_params.copy()

    def evaluate(self, params):
        costs = []
        for _ in range(3):
            solver = BeeTSP(params['num_bees'], params['num_sites'], params['limit'])
            _, history = solver.run()
            costs.append(history[-1])
        return np.mean(costs)

    def tune(self, callback=None):
        changed = True
        cycle = 0
        MAX_CYCLES = 3
        
        if callback: callback(f"Starting tuning... Initial: {self.current_params}\n")

        while changed and cycle < MAX_CYCLES:
            changed = False
            cycle += 1
            if callback: callback(f"--- Cycle {cycle} ---\n")
            
            keys = ['num_bees', 'num_sites', 'limit']
            for key in keys:
                best_val_for_key = self.current_params[key]
                best_cost_for_key = float('inf')
                
                for val in self.param_ranges[key]:
                    test_params = self.current_params.copy()
                    test_params[key] = val
                    
                    cost = self.evaluate(test_params)
                    if callback: callback(f"Testing {key}={val} -> Cost: {cost:.2f}\n")
                    
                    if cost < best_cost_for_key:
                        best_cost_for_key = cost
                        best_val_for_key = val
                
                if best_val_for_key != self.current_params[key]:
                    if callback: callback(f"Update {key}: {self.current_params[key]} -> {best_val_for_key}\n")
                    self.current_params[key] = best_val_for_key
                    changed = True
                    self.best_params = self.current_params.copy()
                else:
                    if callback: callback(f"Keep {key}={best_val_for_key}\n")

        if callback: callback(f"Tuning Optimized. Best Params: {self.best_params}\n")
        return self.best_params
