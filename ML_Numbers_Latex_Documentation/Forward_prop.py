def forwardProp(A , W_1 , b_1, W_2, b_2):
  A_1 = np.dot(W_1.T , A)
  Z_1 = A_1 + b_1
  A_1 = ReLU(Z_1)
  A_2 = np.dot(W_2.T, A_1)
  Z_2 = A_2 + b_2
  A_2 = softmax(Z_2)
  return A_1, Z_1, A_2 , Z_2