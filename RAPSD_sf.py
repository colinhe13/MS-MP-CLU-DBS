import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_rapsd(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算傅里叶变换
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # 计算频域表示的功率谱密度
    power_spectrum_density = np.abs(dft_shift)**2

    # 获取图像的大小
    rows, cols = image.shape

    # 构建频域坐标
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x -= int(cols / 2)
    y -= int(rows / 2)
    radial_frequency = np.round(np.sqrt(x**2 + y**2)).astype(int)

    # 计算径向平均功率谱密度
    rapsd = np.zeros(np.max(radial_frequency))

    for i in range(1, np.max(radial_frequency) + 1):
        rapsd[i - 1] = np.sum(power_spectrum_density[radial_frequency == i]) / np.count_nonzero(radial_frequency == i)

    return rapsd

def plot_rapsd(image_path):
    rapsd = calculate_rapsd(image_path)

    # 构建横坐标 Radial Frequency
    radial_frequency = np.arange(1, len(rapsd) + 1)

    # 将横坐标换算为 Spatial Frequency (cycles/pixel)
    # spatial_frequency = 1 / radial_frequency
    spatial_frequency = radial_frequency / max(radial_frequency)

    # 绘制 RAPSD 图
    plt.plot(spatial_frequency, rapsd)
    plt.xlabel("Spatial Frequency (cycles/pixel)")
    plt.ylabel("RAPSD")
    plt.title("Radially Averaged Power Spectral Density")
    plt.show()

# 请将 "your_image_path.jpg" 替换为你的图片路径
image_path = "output/231208/screen/screen_gray127.png"
plot_rapsd(image_path)
