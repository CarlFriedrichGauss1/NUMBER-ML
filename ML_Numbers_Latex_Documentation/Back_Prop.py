def backProp(A, A_1, A_2, Z_1, Z_2, W_1, W_2, mean):
    vector = np.zeros((10, 1))
    vector[mean, 0] = 1

    
    dZ2 = 2 * (A_2 - vector) 
    dW2 = np.dot(A_1, dZ2.T)
    db2 = dZ2

    dA1 = np.dot(W_2, dZ2) 
    dZ1 = dA1 * d_ReLU(Z_1) 
    dW1 = np.dot(A, dZ1.T) 
    db1 = dZ1
    

    return dW1, db1, dW2, db2