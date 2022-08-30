import timeit
import torch
import numpy as np
import matplotlib.pyplot as plt

runtimes = []
threads = [1] + [t for t in range(2, 25, 2)]
print("Default:", torch.get_num_threads())
for t in threads:
    print("{} thread{}...".format(t, '' if t == 1 else 's'))
    torch.set_num_threads(t)
    r = timeit.timeit(setup = "import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
    runtimes.append(r)

print("Optimal thread count:", threads[np.argmin(runtimes).item()])

plt.plot(threads, runtimes)
plt.show()
