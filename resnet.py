# ResNet - implementation, following Section 4.2 of the ResNet paper, https://arxiv.org/abs/1512.03385
# Jul-Aug 2023 (v1).

import torch
torch.manual_seed(42)

# should report True
torch.cuda.is_available()

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# hyperparameters
num_epochs = 164       # number of training epochs per model (164 = 64k steps)
ilr        = 0.1       # initial learning rate
nseq       = [3,5,7,9] # n determines network depth
batch_size = 128       # batch size on GPU

# the training data undergoes some basic augmentation, but the test data does not
training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Pad(4,fill=128),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
])

train_data = datasets.CIFAR10(
    root="/data/pytorch_data",                   # this is where it will download it to
    train=True,                                  # create dataset from training set
    download=True,                               # download, but only once
    transform=training_transforms,               # takes in a PIL image [0-255] and returns a transformed version [0-1]
)

test_data = datasets.CIFAR10(
    root="/data/pytorch_data",                   # this is where it will download it to
    train=False,                                 # create dataset from test set
    download=True,                               # download, but only once
    transform=transforms.ToTensor(),             # takes in a PIL image and returns a Tensor
)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=batch_size)

# How long are those?
len(train_loader)
len(test_loader)

# grab a training batch
for b in train_loader:
    break

# b is a list of length 2
# b[0] is a Tensor shape (128, 3, 32, 32) which is (batch, channels, h, w)
# b[1] is a Tensor shape (128, ) which is the class 0-9

from matplotlib import pyplot as plt
t = b[0][0]
p = t.permute(1,2,0)
plt.imshow(p, interpolation='nearest')
plt.show()

train_data.classes
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# classes

class Same(nn.Module):
    '''Two layer conv net with optional shortcut connection'''
    def __init__(self, resnet, channels):
        super().__init__()
        self.resnet = resnet
        self.c1     = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding='same')
        self.bn1    = nn.BatchNorm2d(channels)
        self.c2     = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding='same')
        self.bn2    = nn.BatchNorm2d(channels)
        self.relu   = nn.ReLU()
    def forward(self, i):
        x = self.c1(i)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        if self.resnet:
            x = x + i
        x = self.relu(x)
        return x


class Down(nn.Module):
    '''Similar to Same, but doubles the channels and halves the feature map size'''
    def __init__(self, resnet, in_channels):
        super().__init__()
        self.resnet      = resnet
        self.in_channels = in_channels
        self.c1          = nn.Conv2d(in_channels,  in_channels*2,kernel_size=3,stride=2,padding=1)
        self.bn1         = nn.BatchNorm2d(in_channels*2)
        self.c2          = nn.Conv2d(in_channels*2,in_channels*2,kernel_size=3,stride=1,padding='same')
        self.bn2         = nn.BatchNorm2d(in_channels*2)
        self.relu        = nn.ReLU()
    def forward(self, i):
        # i shape                     (batch, in_channels,   feat_map_size,   feat_map_size)
        x = self.c1(i)
        # x shape                     (batch, in_channels*2, feat_map_size/2, feat_map_size/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        if self.resnet:
            x[:, :self.in_channels, :, :] += i[:, :, ::2, ::2]
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    '''CIFAR-10 plain or residual network following Section 4.2 of the ResNet paper'''
    def __init__(self, resnet: bool, n: int):
        super().__init__()
        self.resnet = resnet          # resnet or plain net
        self.n      = n               # number of blocks per out_channel size
        self.md     = nn.ModuleDict() # ordered dict of convolutional layers / blocks
        #
        # the first layer converts 3-channel data into 16-channel data
        self.md.update({'conv1_0' : nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding='same')})
        self.md.update({'relu1_0' : nn.ReLU()})
        #
        # there are then 3n convolution blocks, n with each of {16,32,64} filters:
        #
        # the conv2_i blocks use 16-channels throughout
        for i in range(n):
            self.md.update({f'conv2_{i}' : Same(resnet, channels=16)})
        #
        # the conv3_0 block increases channels from 16 to 32 and downsamples the feature map by 2
        self.md.update({'conv3_0' : Down(resnet, in_channels=16)})
        # subsequent conv3_i block use 32-channels throughout
        for i in range(1,n):
            self.md.update({f'conv3_{i}' : Same(resnet, channels=32)})
        #
        # the conv4_0 block increases channels from 32 to 64 and downsamples the feature map by 2
        self.md.update({'conv4_0' : Down(resnet, in_channels=32)})
        # subsequent conv4_i blocks use 64-channels throughout
        for i in range(1,n):
            self.md.update({f'conv4_{i}' : Same(resnet, channels=64)})
        #
        # the global average pooling layer requires the dynamnic feature map size
        # the network ends with a fully connected layer converting pooled 64-channel data into 10-logits
        self.fc = nn.Linear(64, 10)
        #
    def forward(self, x):
        # the input data x must be 3-channel data because that is hardcoded in the module dict above
        # any feature map size can be used; the following comments assume 32x32 inputs
        # shapes are                             (batch, ch, hi, wi)
        # input x is assumed to be               (batch,  3, 32, 32)
        # 
        # apply the convolutional layers / blocks
        for l in self.md.values():
            x = l(x)
        # output x is expected to be             (batch, 64,  8,  8)
        #
        # the global average pooling layer needs to know the dynamic shape of x
        x = nn.AvgPool2d(x.shape[-2],x.shape[-1])(x)
        # post pooling shape                     (batch, 64,  1,  1)
        # squeeze last two dims
        x = x[:,:,0,0]
        # post squeeze shape                     (batch, 64)
        # fully connected layer
        x = self.fc(x)
        # post fc shape                          (batch, 10)
        # return logits
        return x

    
def train_model(model_type, n, writer, num_epochs, ilr, train_loader, test_loader):
    '''
    train a model of the specified type and depth, writing results to TensorBoard
    '''
    # instantiate a model and move it to the GPU
    model = ResNet(model_type == 'ResNet', n)
    model.cuda()
    #
    layers = len([n for n, p in model.named_parameters() if "bn" not in n])//2
    print('Model type       = ', model_type)
    print('n                = ', n)
    print('Model Layers     = ', layers)
    print('Model Parameters = ', sum([p.numel() for p in model.parameters()]))
    #
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=ilr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs//2,3*num_epochs//4], gamma=0.1, verbose=False)
    #
    # train loop
    steps = 0
    for epoch in range(1,num_epochs+1):
        # train
        _ = model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for b in train_loader:
            images, labels = b
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(images)
            loss   = loss_fn(logits, labels)
            train_loss += loss.item()
            loss.backward()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
            optimizer.step()
            steps += 1
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_err = 100 - train_acc
        writer.add_scalars('Train Error', {f'{layers}': train_err}, steps)
        # test
        _ = model.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            for b in test_loader:
                images, labels = b
                images = images.cuda()
                labels = labels.cuda()
                logits = model(images)
                loss   = loss_fn(logits, labels)
                test_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total   += labels.shape[0]
            test_loss /= len(test_loader)
            test_acc = 100 * correct / total
            test_err = 100 - test_acc
            writer.add_scalars('Test Error', {f'{layers}': test_err}, steps)
        print(f'epoch {epoch:2d} steps {steps:4d} '
              f'Loss : train {train_loss:.3f} test {test_loss:.3f} '
              f'Error(%) : train {train_err:.3f} test {test_err:.3f}')
        scheduler.step()
        

def train_type(model_type, nseq, num_epochs, ilr, train_loader, test_loader):
    '''
    train a sequence of models of the specified type, writing graphs to TensorBoard
    '''
    # Create separate TensorBoard logs for each model type
    # Run Tensorboard with 'tensorboard --logdir=runs'
    # Point browser at http://localhost:6006/
    writer = SummaryWriter(f'runs/{model_type}')
    #
    for n in nseq:
        train_model(model_type, n, writer, num_epochs, ilr, train_loader, test_loader)


# train a sequence of Plain networks and their corresponding ResNets
train_type('Plain',  nseq, num_epochs, ilr, train_loader, test_loader)
train_type('ResNet', nseq, num_epochs, ilr, train_loader, test_loader)

# To colour corresponding networks the same colour in TensorBoard, click on the colour palette
# icon, top left, under Time Series, near Run, and enter (Error_20|Error_32|Error_44|Error_56)
# Then in the Settings column click 'Link by Step' and click at the end of your data for a legend.                                                
