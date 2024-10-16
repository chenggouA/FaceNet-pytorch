import torch
from torch import nn

class conv3x3(nn.Module):
    def __init__(self, in_planes, out_channels, stride=1, padding=0):
        super(conv3x3, self).__init__()
        self.conv3x3 = nn.Sequential(
             nn.Conv2d(in_planes, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),#卷积核为3x3
             nn.BatchNorm2d(out_channels),#BN层，防止过拟合以及梯度爆炸
             nn.ReLU()#激活函数
        )
        
    def forward(self, input):
        return self.conv3x3(input)
        
class conv1x1(nn.Module):
    def __init__(self, in_planes, out_channels, stride=1, padding=0):
        super(conv1x1, self).__init__()
        self.conv1x1 = nn.Sequential(
             nn.Conv2d(in_planes, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False),#卷积核为1x1
             nn.BatchNorm2d(out_channels),
             nn.ReLU()
        )

    def forward(self, input):
        return self.conv1x1(input)


class Inception_ResNet_A(nn.Module):
    '''
        处理后通道数不变
    '''
    def __init__(self, input):
        super(Inception_ResNet_A, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=32,stride=1, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32, stride=1, padding=1)
        self.line =  nn.Conv2d(96, 256, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):

        c1 = self.conv1(x) # 先使用1x1 降维到 32
        c2 = self.conv1(x) # 
        c3 = self.conv1(x)
        c2_1 = self.conv2(c2)
        c3_1 = self.conv2(c3)
        c3_2 = self.conv2(c3_1) # 两个3x3 卷积 替代 5x5卷积
        cat = torch.cat([c1, c2_1, c3_2],dim=1) # torch.Size([b, 3x32, h, w])
        line = self.line(cat) # 使用1x1卷积调整通道数
        out = x + line
        out = self.relu(out)

        return out



class Reduction_A(nn.Module):
    '''
        处理后宽高减半，通道增加
    '''
    def __init__(self, input, n=384, k=192, l=224, m=256):
        super(Reduction_A, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv1 = conv3x3(in_planes=input, out_channels=n, stride=2, padding=0)
        self.conv2 = conv1x1(in_planes=input, out_channels=k, padding=1)
        self.conv3 = conv3x3(in_planes=k, out_channels=l,padding=0)
        self.conv4 = conv3x3(in_planes=l, out_channels=m,stride=2,padding=0)

    def forward(self, x):

        c1 = self.maxpool(x) # stride = 2, 宽高减半
        c2 = self.conv1(x) # stride = 2, 宽高减半
        c3 = self.conv2(x)
        c3_1 = self.conv3(c3)
        c3_2 = self.conv4(c3_1)
        cat = torch.cat([c1, c2, c3_2], dim=1) # b, input + n + m, h / 2, w / 2 

        return cat



class Inception_ResNet_B(nn.Module):
    '''
        使用非对称的1x7, 7x1 卷积
    '''
    def __init__(self, input):
        super(Inception_ResNet_B, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=128, stride=1, padding=0)
        self.conv1x7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,7), padding=(0,3))
        self.conv7x1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(7,1), padding=(3,0))
        self.line = nn.Conv2d(256, 896, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):

        c1 = self.conv1(x)
        c2 = self.conv1(x)
        c2_1 = self.conv1x7(c2)

        c2_1 = self.relu(c2_1) # 非对称卷积

        c2_2 = self.conv7x1(c2_1)

        c2_2 = self.relu(c2_2)

        cat = torch.cat([c1, c2_2], dim=1)
        line = self.line(cat)
        out = x + line

        out = self.relu(out)
        return out



class Reduction_B(nn.Module):
    def __init__(self, input):
        super(Reduction_B, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = conv1x1(in_planes=input, out_channels=256, padding=1)
        self.conv2 = conv3x3(in_planes=256, out_channels=384, stride=2, padding=0)
        self.conv3 = conv3x3(in_planes=256, out_channels=256, stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=256, out_channels=256, padding=1)
        self.conv5 = conv3x3(in_planes=256, out_channels=256, stride=2, padding=0)
    
    def forward(self, x):

        c1 = self.maxpool(x)
        c2 = self.conv1(x)
        c3 = self.conv1(x)
        c4 = self.conv1(x)
        c2_1 = self.conv2(c2)
        c3_1 = self.conv3(c3)
        c4_1 = self.conv4(c4)
        c4_2 = self.conv5(c4_1)
        cat = torch.cat([c1, c2_1, c3_1,c4_2], dim=1)
        return cat
    

class Inception_ResNet_C(nn.Module):
    '''
        使用1x3, 3x1的非对称卷积
    '''
    def __init__(self, input):
        super(Inception_ResNet_C, self).__init__()
        self.conv1 = conv1x1(in_planes=input, out_channels=192, stride=1, padding=0)
        self.conv1x3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 3), padding=(0,1))
        self.conv3x1 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 1), padding=(1,0))
        self.line = nn.Conv2d(384, 1792, 1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv1(x)
        c2_1 = self.conv1x3(c2)

        c2_1 = self.relu(c2_1)
        c2_2 = self.conv3x1(c2_1)

        c2_2 = self.relu(c2_2)
        cat = torch.cat([c1, c2_2], dim=1)
        line = self.line(cat)
        out = x + line
        out = self.relu(out)

        return out


class StemV1(nn.Module):
    def __init__(self, in_planes):
        super(StemV1, self).__init__()

        self.conv1 = conv3x3(in_planes=in_planes, out_channels=32,stride=2, padding=0)
        self.conv2 = conv3x3(in_planes=32, out_channels=32, stride=1, padding=0)
        self.conv3 = conv3x3(in_planes=32, out_channels=64, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv4 = conv3x3(in_planes=64, out_channels=64, stride=1, padding=1)
        self.conv5 = conv1x1(in_planes=64, out_channels=80, stride=1, padding=0)
        self.conv6 = conv3x3(in_planes=80, out_channels=192, stride=1, padding=0)
        self.conv7 = conv3x3(in_planes=192, out_channels=256, stride=2, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x



class Inception_ResNet(nn.Module):
    def __init__(self, classes=2):
        super(Inception_ResNet, self).__init__()
        blocks = []
        blocks.append(StemV1(in_planes=3))
        for _ in range(5):
            blocks.append(Inception_ResNet_A(input=256))
        blocks.append(Reduction_A(input=256))
        for _ in range(10):
            blocks.append(Inception_ResNet_B(input=896))
        blocks.append(Reduction_B(input=896))
        for _ in range(10):
            blocks.append(Inception_ResNet_C(input=1792))

        self.features = nn.Sequential(*blocks)


    def forward(self, x):
        x = self.features(x)

        return x



if __name__ == "__main__":
    x = torch.randn((1, 3, 299, 299))

    model = Inception_ResNet(1000)

    output = model(x)
    print(output.shape)






