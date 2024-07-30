import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

neg_log = -np.log(softmax_outputs[ range(len(softmax_outputs)), class_targets])

avg_loss = np.mean(neg_log)

print(avg_loss)

# Dynamic for sparse(1D) or one-hot-encode(2D)

class_targets = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]])

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)

#Loss
neg_log = -np.log(correct_confidences)

avg_loss = np.mean(neg_log)

print(avg_loss)

#Clipping for log(0) error
# y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) 
