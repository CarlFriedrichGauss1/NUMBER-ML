def gradientDescent(W_1, b_1, W_2, b_2, learning_rate, epochs):
    accuracy_list = []
    
    for epoch in range(epochs):
      correct_predictions = 0
      total_samples = len(x_train)
      for i in range(total_samples):
          A = A_layers[i]
          A_1, Z_1, A_2 , Z_2 = forwardProp(A, W_1, b_1, W_2, b_2)

          predicted_label = np.argmax(A_2)
          if predicted_label == y_train[i]:
              correct_predictions += 1

          dW_1, db_1, dW_2, db_2 = backProp(A, A_1, A_2, Z_1, Z_2, W_1, W_2 ,y_train[i])

          W_1 = W_1 - learning_rate * dW_1
          b_1 = b_1 - learning_rate * db_1
          W_2 = W_2 - learning_rate * dW_2
          b_2 = b_2 - learning_rate * db_2
        
      accuracy = (correct_predictions / total_samples) * 100
      accuracy_list.append(accuracy)
      print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")

      if (epoch + 1) % 5 == 0:
        learning_rate *= 0.5
        print(f"Learning rate decayed to {learning_rate}")

    return accuracy_list[-1], W_1, b_1, W_2, b_2