# check if random images belong to correct classes
fig=plt.figure(figsize=(9,9))
rows,cols=4,4
for i in range(1,rows*cols+1):
  random_idx=torch.randint(0,len(test_dataset),size=[1]).item()
  model.eval()
  with torch.inference_mode():
    img=(test_dataset[random_idx][0]).unsqueeze(dim=0).to(device)
    y_pred=model(img).argmax(dim=1).item()
  y_true=test_dataset[random_idx][1]

  fig.add_subplot(rows,cols,i)
  img=img.squeeze(dim=0).permute(1,2,0).cpu()
  plt.axis(False)
  plt.imshow(img)
  key = next(key for key, value in label_classes.items() if value == y_pred)

  if y_true==y_pred:
    plt.title(key,c="g")
  else:
    plt.title(key,c="r")
