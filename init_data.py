import torch
import numpy as np
from configs.path_internal import project_root_path
from torchvision import datasets, transforms

transform = transforms.ToTensor()


# 定义保存数据的函数
def save_mnist_data(loader, filename):
    images, labels = next(iter(loader))
    np_images = images.numpy()  # 将 Tensor 转换为 NumPy 数组
    np_labels = labels.numpy()  # 将 Tensor 转换为 NumPy 数组

    # 保存图像数据到 CSV 文件（每个像素值用逗号分隔，每张图片占一行）
    np.savetxt(f'{filename}_images.csv', np_images.reshape(images.size(0), -1), delimiter=',')

    # 保存标签数据到 CSV 文件
    np.savetxt(f'{filename}_labels.csv', np_labels, delimiter=',')


def get_mnist_data():
    train_set = datasets.MNIST(root=f'{project_root_path}/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = datasets.MNIST(root=f'{project_root_path}/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    save_mnist_data(trainloader, f'{project_root_path}/data/MNIST/train')
    save_mnist_data(testloader, f'{project_root_path}/data/MNIST/test')
    return


def run():
    get_mnist_data()
    return


if __name__ == '__main__':
    run()
