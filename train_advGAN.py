import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from train_target_model import target_net, input_size

use_pretrained=True
feature_extract_only=True
target_name = 'densenet'  # densenet169  or resnet50
extractor_name = 'resnet'  # densnet121  or resnet18
num_classes = 10
num_features = 10
image_nc= 3
epochs = 60
batch_size = 300
BOX_MIN = 0
BOX_MAX = 256
nz = 30
# train advGAN

if __name__ == "__main__":
    # Define what device we are using
    use_cuda = True
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    # target_model
    pretrained_model = "./target_model.pth"
    target_modeled, input_size = target_net(target_name,
                                            num_classes,
                                            feature_extract_only,
                                            use_pretrained).initialize_model()
    targeted_model = target_modeled.to(device)
    targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model.eval()
    # feature extractor
    pretrained_model = "./feature_extract.pth"
    feature_extract_net, feature_extract_input_size = target_net(extractor_name,
                                                                 num_features,
                                                                 feature_extract_only,
                                                                 use_pretrained).initialize_model()
    feature_extract_net = feature_extract_net.to(device)
    feature_extract_net.load_state_dict(torch.load(pretrained_model))
    feature_extract_net.eval()


    # train dataset and dataloader declaration
    train_data_transform = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = torchvision.datasets.CIFAR10('./dataset', train=True, transform=train_data_transform, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    advGAN = AdvGAN_Attack(device, targeted_model, feature_extract_net, num_classes, num_features, BOX_MIN, BOX_MAX, nz, batch_size)
    advGAN.train(dataloader, epochs)
