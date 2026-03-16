import time
from qpbenchmark.runner import BenchmarkRunner

def dummy_generator(n):
    for i in range(n):
        yield ("solve", type("Problem", (), {"name": f"P{i}"})(), "solver", "setting", {})

class DummyResults:
    def update(self, *args):
        pass
    def write(self):
        pass

stats = {"solve": 0, "skip": 0, "skip_timeout": 0, "record_failure": 0}
runner = BenchmarkRunner(max_workers=1, verbose=True)

t0 = time.time()
runner.execute(dummy_generator(100), 100, DummyResults(), stats)
t1 = time.time()
print(f"Throughput: {100 / (t1 - t0):.2f} it/s")
