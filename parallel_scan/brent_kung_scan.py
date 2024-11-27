import numpy as np


def brent_kung_scan(x):
    n = len(x)
    y = x.copy()

    # Upsweep phase: builds binary tree from bottom to top
    # stride doubles each time: 1 -> 2 -> 4 -> 8 -> ... -> n/2
    stride = 1
    while stride < n:
        for i in range(n):
            index = (i + 1) * 2 * stride - 1
            if index < n:
                y[index] += y[index - stride]
        stride *= 2
    
    # Downsweep phase: distributes values down the tree
    # starts from n//4 (not n/2) because:
    # - last element at stride n/2 already has correct value
    # - need to start from middle level of tree
    # stride halves each time: n/4 -> n/8 -> n/16 -> ... -> 1
    stride = n // 4
    while stride > 0:
        for i in range(n):
            index = (i + 1) * stride * 2 - 1
            if index + stride < n:
                y[index + stride] += y[index]
        stride //= 2

    return y
    
N = 32
x = np.array([i for i in range(N)], dtype=np.float32)
y = brent_kung_scan(x)

print("Input: ", x)
print("Output:", y)
