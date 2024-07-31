import numpy as np
import matplotlib.pyplot as plt

# def f(x):
#     return 2*x


# x = np.array(range(5))
# y = f(x)

# print(x)
# print(y)

# plt.plot(x, y)
# plt.show()

#Slope for liner func
# slope = ((y[1]-y[0]) / (x[1]-x[0]))
# print(slope)
# slope2 = ((y[3]-y[2]) / (x[3]-x[2]))
# print(slope2)

def f(x):
    return 2*x**2

x = np.arange(0,5,0.001)
y = f(x)

plt.plot(x, y)
# plt.show()

colors = ['k','g','r','b','c']

def approx_tangent_line(x, approx_deriv):
    return (approx_deriv*x) + b


for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2= x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1, y1), (x2, y2))
    
    approx_deriv = (y2-y1)/(x2-x1)
    b = y2 - approx_deriv*x2
    



    to_plot = [x1-0.9, x1, x1+0.9]
    plt.scatter(x1, y1, c=colors[i])
    plt.plot([point for point in to_plot], 
             [approx_tangent_line(point, approx_deriv) for point in to_plot], c=colors[i])
    
    print('Approximate derivative for f(x)',
    f'where x = {x1} is {approx_deriv}')
    
plt.show()