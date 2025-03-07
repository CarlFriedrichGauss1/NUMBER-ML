def ReLU(x):
  return np.maximum(0, x)

def softmax(Z):
  Z = Z - np.max(Z)  
  exp_values = np.exp(Z)
  sum_exp_values = np.sum(exp_values)
  return A_out_prob

def d_ReLU(x):
  return x > 0