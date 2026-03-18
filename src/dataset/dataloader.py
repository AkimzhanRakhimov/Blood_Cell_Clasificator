# transform img to tensor
transform=transforms.Compose([transforms.Resize((240,320)),
                             transforms.ToTensor()])
dataset=CellDataset(dataset_path,label_classes,transform)
# divide dataset into train and test subsets
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_dataset,test_dataset=random_split(dataset,[train_size,test_size])
# create dataloader
train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=4,pin_memory=True)
test_dataloader=DataLoader(test_dataset,shuffle=True,batch_size=BATCH_SIZE,num_workers=4,pin_memory=True)
