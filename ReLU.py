import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output1 = []

# Approach-1
for i in inputs:
    if i > 0:
        output1.append(i)
    else:
        output1.append(0)
print(output1)

output2 = []
# Approach-2
for i in inputs:
    output2.append(max(0, i))
print(output2)

# Approach-3
output3 = np.maximum(0, inputs)

print(output3)
