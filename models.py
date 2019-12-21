import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,models,transforms

class target_net():
    def __init__(self, model_name, num_classes, feature_extract, use_pretrained):
        super(target_net, self).__init__()  # MNIST:1*28*28  cifar:3*32*32
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False  # 提取特征或变动所有权重

    def initialize_model(self):
        model_ft = None
        input_size = 0

        if self.model_name == 'resnet':
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'alexnet':
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'vgg':
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'squeezenet':
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == 'densenet':
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == 'inception':
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.Auxlogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 299

        else:
            print('invalid model name,exiting...')
            exit()

        return model_ft, input_size



class feature_extract_net():
    def __init__(self, model_name, num_features, feature_extract, use_pretrained):
        super(feature_extract_net, self).__init__()  # MNIST:1*28*28  cifar:3*32*32
        self.model_name = model_name
        self.num_features = num_features  # 当使用蒸馏模型时与num_classes一样
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained


    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False  # 提取特征或变动所有权重

    def initialize_model(self):
        model_ft = None
        input_size = 0

        if self.model_name == 'resnet':
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_features)
            input_size = 224

        elif self.model_name == 'alexnet':
            model_ft = models.alexnet(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_features)
            input_size = 224

        elif self.model_name == 'vgg':
            model_ft = models.vgg11_bn(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.num_features)
            input_size = 224

        elif self.model_name == 'squeezenet':
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_features, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = self.num_features
            input_size = 224

        elif self.model_name == 'densenet':
            model_ft = models.densenet121(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_features)
            input_size = 224

        elif self.model_name == 'inception':
            model_ft = models.inception_v3(pretrained=self.use_pretrained)
            self.set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.Auxlogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_features)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_features)
            input_size = 299

        else:
            print('invalid model name,exiting...')
            exit()

        return model_ft, input_size


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # MNIST:1*28*28  cifar:3*32*32
        model = [
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13  cifar:8*15*15
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5   cifar:16*6*6
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 2, 2),
            nn.Sigmoid()
            # 32*1*1  cifar:32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output



class Generator(nn.Module):
    def __init__(self, nz, num_features):
        super(Generator, self).__init__()
        self.nz=nz
        self.num_features=num_features

        self.l1 = nn.Sequential(nn.Linear(self.num_features + self.nz, 3 * 8 * 2 * 2))
        # state size. (ngf*8) x 2 x 2

        self.main1 = nn.Sequential(nn.ConvTranspose2d(3 * 8, 3 * 8, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(3 * 8),nn.ReLU(True),
                                   # state size. (ngf*8) x 4 x 4
                                   nn.ConvTranspose2d(3 * 8, 3 * 4, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(3 * 4),nn.ReLU(True),
                                   # state size. (ngf*4) x 8 x 8
                                   nn.ConvTranspose2d( 3 * 4, 3 * 2, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(3 * 2),nn.ReLU(True),
                                   # state size. (ngf*2) x 16 x 16
                                   nn.ConvTranspose2d( 3 * 2, 3, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(3),nn.Tanh()
                                   # state size. (ngf) x 32 x 32
                                   )

    def forward(self, input):
        out = self.l1(input)
        out = out.reshape(out.shape[0], 3 * 8, 2, 2)
        out = self.main1(out)
        return out