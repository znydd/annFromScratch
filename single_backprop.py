x = [1.0, -2.0, 3.0] #input values
w = [-3.0, -1.0, 2.0] #weights
b = 1.0 #bias

# Multiplying inputs with weights
xw0 = x[0] * w[0]  
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)

# adding bias
z = xw0 + xw1 + xw2 + b
print(z)

# ReLU activation function
y = max(z, 0)


# backword pass

# assume the derivative from the next layer
dvalue = 1

#deriv of ReLU
drelu_dz = dvalue * (1.0 if z > 0 else 0.0) # we are multiplyting because in chain rule you have to multiply with the prev deirv
print(drelu_dz)

# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


#caluculated gradients
dx = [drelu_dx0, drelu_dx1, dmul_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dxw2]
db = drelu_db

#current weights and bias
print(w, b)
#updating the weight
w[0] = w[0] - 0.001 * dw[0]
w[1] = w[1] - 0.001 * dw[1]
w[2] = w[2] - 0.001 * dw[2]
b = b - 0.001 * db
print(w, b)


# now if we do forward pass

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
# Adding
z1 = xw0 + xw1 + xw2 + b
# ReLU activation function
y = max(z1, 0)
print(y) # we just decreacsed that from 6 to 5.9

