# training and evaluating loop
epochs=10

for epoch in range(epochs):
  model.train()
  train_loss,train_acc=0,0
  print(f"Epoch: {epoch+1}/{epochs}")
  for batch,(X,y_true) in enumerate(train_dataloader):
    X,y_true=X.to(device),y_true.to(device)
    y_pred=model(X)
    loss=loss_fn(y_pred,y_true)
    train_loss+=loss
    train_acc+=accuracy_fn(y_pred.argmax(dim=1),y_true)
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if batch%20==0:
      print(f"Loooked at {batch*BATCH_SIZE}/{len(train_dataloader)*BATCH_SIZE} samples")
      print(f"Predicted labels: {y_pred.argmax(dim=1)}")
      print(f"True labels: {y_true}")
  train_loss/=len(train_dataloader)
  train_acc/=len(train_dataloader)
  test_loss,test_acc=0,0
  model.eval()
  with torch.inference_mode():
    for batch,(X_test,y_test_true) in enumerate(test_dataloader):
      X_test,y_test_true=X_test.to(device),y_test_true.to(device)
      y_test_pred=model(X_test)
      test_loss+=loss_fn(y_test_pred,y_test_true)
      test_acc+=accuracy_fn(y_test_pred.argmax(dim=1),y_test_true)
    test_loss/=len(test_dataloader)
    test_acc/=len(test_dataloader)
  print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
