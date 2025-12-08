from solver import ParameterTuner
import time

def log(msg):
    print(msg, end='')

if __name__ == "__main__":
    tuner = ParameterTuner()
    print("Starting tuning...")
    start = time.time()
    best_params = tuner.tune(log)
    end = time.time()
    print(f"\nTuning finished in {end - start:.2f} seconds.")
    print(f"Best Params: {best_params}")
