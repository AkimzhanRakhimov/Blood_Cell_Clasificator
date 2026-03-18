# custom accuracy
def accuracy_fn(y_pred,y_true):
  correct=torch.eq(y_pred,y_true).sum().item()
  acc=(correct/len(y_true))*100
  return acc
