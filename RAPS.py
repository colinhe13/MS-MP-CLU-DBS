import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot


def calculate_raps(image):
    # 进行傅里叶变换
    fft_image = np.fft.fft2(image)

    # 计算功率谱
    power_spectrum = np.abs(fft_image) ** 2
    # 计算功率谱并进行对数变换
    # power_spectrum = np.log(1 + np.abs(fft_image) ** 2)

    # 计算频率
    freq_x = np.fft.fftfreq(image.shape[0])
    freq_y = np.fft.fftfreq(image.shape[1])

    # 将频率表示为极坐标
    freq_r, freq_theta = np.meshgrid(np.sqrt(freq_x ** 2 + freq_y ** 2), np.arctan2(freq_y, freq_x))

    # 将频率表示为径向坐标
    freq_r_bins = np.linspace(0.0, 0.5, min(image.shape) // 2)  # 使用图像尺寸的一半作为径向坐标的数量
    raps, _ = np.histogram(freq_r, bins=freq_r_bins, weights=power_spectrum)
    counts, _ = np.histogram(freq_r, bins=freq_r_bins)

    # 避免除以零
    mask = counts != 0
    raps[mask] /= counts[mask]

    return freq_r_bins[:-1], raps


def plot_raps(freq_r, raps):
    # 绘制径向平均功率谱
    plt.plot(freq_r, raps)
    plt.xlabel('Spatial Frequency (cycles per pixel)')
    plt.ylabel('RAPS')
    plt.title('Radially Averaged Power Spectrum')
    plt.show()


def AnalyzeNoiseTexture(Texture, SingleFigure=True, SimpleLabels=False):
    """Given a 2D array of real noise values this function creates one or more 
       figures with plots that allow you to analyze it, especially with respect to 
       blue noise characteristics. The analysis includes the absolute value of the 
       Fourier transform, the power distribution in radial frequency bands and an 
       analysis of directional isotropy.
      \param A two-dimensional array.
      \param SingleFigure If this is True, all plots are shown in a single figure, 
             which is useful for on-screen display. Otherwise one figure per plot 
             is created.
      \param SimpleLabels Pass True to get axis labels that fit into the context of 
             the blog post without further explanation.
      \return A list of all created figures.
      \note For the plots to show you have to invoke pyplot.show()."""
    FigureList = list()
    if (SingleFigure):
        Figure = pyplot.figure()
        FigureList.append(Figure)

    def PrepareAxes(iAxes, **KeywordArguments):
        if (SingleFigure):
            return Figure.add_subplot(2, 2, iAxes, **KeywordArguments)
        else:
            NewFigure = pyplot.figure()
            FigureList.append(NewFigure)
            return NewFigure.add_subplot(1, 1, 1, **KeywordArguments)

    # Plot the dither array itself
    PrepareAxes(1, title="Blue noise dither array")
    pyplot.imshow(Texture.real, cmap="gray", interpolation="nearest")

    # Plot the Fourier transform with frequency zero shifted to the center
    PrepareAxes(2, title="Fourier transform (absolute value)", xlabel="$\\omega_x$", ylabel="$\\omega_y$")
    DFT = np.fft.fftshift(np.fft.fft2(Texture)) / float(np.size(Texture))
    Height, Width = Texture.shape
    ShiftY, ShiftX = (int(Height / 2), int(Width / 2))
    pyplot.imshow(np.abs(DFT), cmap="viridis", interpolation="nearest", vmin=0.0, vmax=np.percentile(np.abs(DFT), 99),
                  extent=(-ShiftX - 0.5, Width - ShiftX - 0.5, -ShiftY + 0.5, Height - ShiftY + 0.5))
    pyplot.colorbar()

    # Plot the distribution of power over radial frequency bands
    PrepareAxes(3, title="Radial power distribution",
                xlabel="Distance from center / pixels" if SimpleLabels else "$\\sqrt{\\omega_x^2+\\omega_y^2}$")
    X, Y = np.meshgrid(range(DFT.shape[1]), range(DFT.shape[0]))
    X -= int(DFT.shape[1] / 2)
    Y -= int(DFT.shape[0] / 2)
    RadialFrequency = np.asarray(np.round(np.sqrt(X ** 2 + Y ** 2)), dtype=int)
    RadialPower = np.zeros((np.max(RadialFrequency) - 1,))
    DFT[int(DFT.shape[0] / 2), int(DFT.shape[1] / 2)] = 0.0
    for i in range(RadialPower.shape[0]):
        RadialPower[i] = np.sum(np.where(RadialFrequency == i, np.abs(DFT), 0.0)) / np.count_nonzero(
            RadialFrequency == i)
    pyplot.plot(np.arange(np.max(RadialFrequency) - 1) + 0.5, RadialPower)

    # Plot the distribution of power over angular frequency ranges
    PrepareAxes(4, title="Anisotropy (angular power distribution)", aspect="equal",
                xlabel="Frequency x" if SimpleLabels else "$\\omega_x$",
                ylabel="Frequency y" if SimpleLabels else "$\\omega_y$")
    CircularMask = np.logical_and(0 < RadialFrequency, RadialFrequency < int(min(DFT.shape[0], DFT.shape[1]) / 2))
    NormalizedX = np.asarray(X, dtype=float) / np.maximum(1.0, np.sqrt(X ** 2 + Y ** 2))
    NormalizedY = np.asarray(Y, dtype=float) / np.maximum(1.0, np.sqrt(X ** 2 + Y ** 2))
    BinningAngle = np.linspace(0.0, 2.0 * np.pi, 33)
    AngularPower = np.zeros_like(BinningAngle)
    for i, Angle in enumerate(BinningAngle):
        DotProduct = NormalizedX * np.cos(Angle) + NormalizedY * np.sin(Angle)
        FullMask = np.logical_and(CircularMask, DotProduct >= np.cos(np.pi / 32.0))
        AngularPower[i] = np.sum(np.where(FullMask, np.abs(DFT), 0.0)) / np.count_nonzero(FullMask)
    MeanAngularPower = np.mean(AngularPower[1:])
    DenseAngle = np.linspace(0.0, 2.0 * np.pi, 256)
    pyplot.plot(np.cos(DenseAngle) * MeanAngularPower, np.sin(DenseAngle) * MeanAngularPower, color=(0.7, 0.7, 0.7))
    pyplot.plot(np.cos(BinningAngle) * AngularPower, np.sin(BinningAngle) * AngularPower)

    return FigureList

def GetRAPS(Texture, SingleFigure=True, SimpleLabels=False):
    FigureList = list()
    if (SingleFigure):
        Figure = pyplot.figure()
        FigureList.append(Figure)

    def PrepareAxes(iAxes, **KeywordArguments):
        if (SingleFigure):
            return Figure.add_subplot(2, 2, iAxes, **KeywordArguments)
        else:
            NewFigure = pyplot.figure()
            FigureList.append(NewFigure)
            return NewFigure.add_subplot(1, 1, 1, **KeywordArguments)

    PrepareAxes(3, title="Radial power distribution",
                xlabel="Distance from center / pixels" if SimpleLabels else "$\\sqrt{\\omega_x^2+\\omega_y^2}$")
    DFT = np.fft.fftshift(np.fft.fft2(Texture)) / float(np.size(Texture))
    X, Y = np.meshgrid(range(DFT.shape[1]), range(DFT.shape[0]))
    X -= int(DFT.shape[1] / 2)
    Y -= int(DFT.shape[0] / 2)
    RadialFrequency = np.asarray(np.round(np.sqrt(X ** 2 + Y ** 2)), dtype=int)
    RadialPower = np.zeros((np.max(RadialFrequency) - 1,))
    DFT[int(DFT.shape[0] / 2), int(DFT.shape[1] / 2)] = 0.0
    for i in range(RadialPower.shape[0]):
        RadialPower[i] = np.sum(np.where(RadialFrequency == i, np.abs(DFT), 0.0)) / np.count_nonzero(
            RadialFrequency == i)
    pyplot.plot(np.arange(np.max(RadialFrequency) - 1) + 0.5, RadialPower)




# 读取图像
# image = np.load("output/seed/seed_7_size_512_bossbase.npy")
# image = np.where(image, 0, 255)
image = cv2.imread("output/231208/mpcludbs_d13_17_p5_h12_gray127.png", cv2.IMREAD_GRAYSCALE)

# cv2.imshow("img", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 计算径向平均功率谱
# freq_r, raps = calculate_raps(image)
# 绘制径向平均功率谱
# plot_raps(freq_r, raps)

# 分析噪声纹理
# figureList = AnalyzeNoiseTexture(image, SingleFigure=False, SimpleLabels=True)
# pyplot.show()

GetRAPS(image, SingleFigure=False, SimpleLabels=True)
pyplot.show()