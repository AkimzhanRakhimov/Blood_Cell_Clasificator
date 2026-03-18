# show image
fig=plt.figure(figsize=(9,9))
rows,cols=4,4
for i in range(1,rows*cols+1):
  random_idx=torch.randint(0,len(dataset),size=[1]).item()
  img=dataset[random_idx][0].permute(1,2,0)
  fig.add_subplot(rows,cols,i)
  plt.axis(False)
  plt.imshow(img)
  key = next(key for key, value in label_classes.items() if value == dataset[random_idx][1])
  plt.title(key)

