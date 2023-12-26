import numpy as np
import scipy.signal as signal
import cv2
import os


def read_and_show(path):
    # img = cv2.imread(path)
    img = np.load(path)
    # img = np.where(img == 0, 0.0, 255.0)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_gray_127():
    # 生成大小为size的灰度级127/255的图像
    size = 256
    img = np.full((size, size), 127, dtype=np.uint8)
    np.save("output/screen/gray127_s256.npy", img)
    return


def cal_cpp(d):
    fs = 13
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            # 计算到中心的距离的二范数
            dist = np.linalg.norm([k, l])
            c = dist ** 2 / (2 * d ** 2)
            # c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
    return GF


def cal_cpp1(d):
    fs = 7
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            # 计算到中心的距离的二范数
            dist = np.linalg.norm([k, l])
            # c = dist**2 / (2 * d ** 2)
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
    return GF

def get_graydiff():
    # 计算灰度值
    img1 = cv2.imread("output/231208/msmpcludbs_d13_17_s5_p10_gray127.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("output/231208/msmpcludbs_d13_17_s5_p10_gray127_1.png", cv2.IMREAD_GRAYSCALE)
    count1 = np.count_nonzero(img1)
    count2 = np.count_nonzero(img2)
    print(256 - count1 / 256, 256 - count2 / 256)
    print(1 - count1 / (256 * 256), 1 - count2 / (256 * 256))

    # for i in range(0, 256):
    #     img = cv2.imread("output/231208/screen/screen_gray" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
    #     count = np.count_nonzero(img)
    #     print("gray " + str(i) + " -- " + str(255 - count*255/(256*256)))

    # 展示两张图像的差异
    # h, w = img1.shape
    # for i in range(h):
    #     for j in range(w):
    #         if img1[i, j] != img2[i, j]:
    #             img1[i, j] = 0
    #         else:
    #             img1[i, j] = 255
    # cv2.imshow("img", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def gen_eps(img, d, d1, cur_values, target_values):
    height, width = img.shape
    CPP = cal_cpp(d)
    CPP1 = cal_cpp(d1)
    HalfCPPSize = 12

    im = np.full((height, width), target_values, dtype=np.float64)
    imr = im / 255.0

    dst = img <= cur_values
    dst = np.where(dst, 1.0, 0.0)
    Err0 = dst - np.full((height, width), target_values / 255.0, dtype=np.float64)
    D_CEP0 = signal.correlate2d(Err0, CPP - CPP1, mode='full')

    Err = dst - imr
    CEP = signal.correlate2d(Err, CPP, mode='full')

    EPS = np.zeros((height, width), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            if dst[i, j] == 0:
                a0 = 1
            else:
                a0 = -1
            EPS[i,j] = a0 * a0 * CPP1[HalfCPPSize, HalfCPPSize] + \
                  2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                  2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j]

    return EPS

# screen = cv2.imread("output/231208/screen/screen_gray127.png", cv2.IMREAD_GRAYSCALE)
# screen = np.where(screen < 127, 127, 255)
# gen_eps(screen, 1.3, 1.7, 127, 128)

# get_graydiff()


# notification_script = """
# display notification "程序运行结束，请查看" with title "Pycharm" sound name "Glass"
# """
# notification_script = """
# display alert "提示" message "content"
# """
# os.system(f"osascript -e '{notification_script}'")


# matrx = np.load("output/231208/screen/screen_matrx.npy")
# img = cv2.imread("resource/bell_b.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("resource/ctrf_sq1.jpg", cv2.IMREAD_GRAYSCALE)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if matrx[i % 256, j % 256] > img[i, j]:
#             img[i, j] = 0.0
#         else:
#             img[i, j] = 255.0
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# generate_gray_127()
# read_and_show("output/231208/screen_gray127.npy")

# for i in range(-1, 2):
#     print(i)
# print(cal_cpp(1.3) == cal_cpp(1.3))
# img = cv2.imread("F:/Ander/Downloads/BossBase-1.01-cover/2.pgm", cv2.IMREAD_GRAYSCALE)
# pgm1 = np.load("output/init11_pd01/msmpcludbs_seed7_d13_17_s5_p10_gray127_0w1b_stage0.npy")
# pgm2 = np.load("output/screen/screen127_seed7_init8_d13_17_s5_p5_gray127.npy")
# pgm1 = np.where(pgm1, 0.0, 1.0)
# pgm2 = np.where(pgm2, 0.0, 1.0)
# cv2.imwrite("output/bb/1.pgm", pgm1 * 255)
# print(pgm1)

# cv2.imshow("pgm1", pgm1 * 255)
# cv2.imshow("pgm2", pgm2 * 255)

# save_filepath = "resource/rev/1.npy"
# np.save(save_filepath, pgm1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

