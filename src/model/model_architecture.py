# neural network consists of 2 convolutional blocks and 1 linear layer
class BloodCellClassifierV0(nn.Module):
  def __init__(self,input_features:int,hidden_dim:int,output_features:int):
    super().__init__()
    self.conv_block_1=nn.Sequential(
        nn.Conv2d(in_channels=input_features,
                  out_channels=hidden_dim,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_dim,
                  out_channels=hidden_dim,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2))
    )
    self.conv_block_2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_dim,
                  out_channels=hidden_dim,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_dim,
                  out_channels=hidden_dim,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2))
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(in_features=hidden_dim*4800,
                  out_features=output_features),
    )

  def forward(self,x):
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))
# create model instance
model=BloodCellClassifierV0(3,10,len(label_classes))
model.to(device)
