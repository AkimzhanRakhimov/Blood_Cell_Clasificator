# preparing confusion matrix
y_true=[]
y_preds=[]

model.eval()
with torch.inference_mode():
  for X,y in test_dataloader:
    X,y=X.to(device),y.to(device)
    y_true.append(y)
    y_pred=model(X).argmax(dim=1)
    y_preds.append(y_pred)
    # print(y_pred)
  y_true_tensor=torch.cat(y_true)
  y_pred_tensor=torch.cat(y_preds)
# showing a heatmap
data=np.zeros((4,4),dtype=int)

# counting true and false predictions
for idx,i in enumerate(y_true_tensor):
  data[i][(y_pred_tensor[idx])]+=1
labels=label_classes.keys()
plt.figure(figsize=(7,5))
# plotting the heatmap
hm = sns.heatmap(data=data,
                annot=True,xticklabels=labels,yticklabels=labels,fmt="d")

# displaying the plotted heatmap
plt.show()
