import numpy as np
import cv2
import scipy.signal as signal


# 产生种子半色调图像的函数
# img: 原图像
# d: 滤波器参数
# g: 种子半色调的灰度级数 0-255/255
def seed(img, d, g):
    fs = 7
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)

    CPP = np.zeros((13, 13))
    HalfCPPSize = 6
    CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))

    im = np.array(img, dtype=np.float64)

    rows, cols = im.shape

    imr = np.random.rand(rows, cols) > g
    # imr = np.random.rand(rows, cols) > 0.02968627451
    imr = np.where(imr, 1.0, 0.0)

    dst = np.random.rand(rows, cols) > 0.5
    dst = np.where(dst, 1.0, 0.0)

    Err = dst - imr

    CEP = signal.correlate2d(Err, CPP, mode='full')
    ESP_MIN = 0

    CountOpp = 0

    while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
        CountB = 0
        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):  # -1 0 1
                    if not (0 <= i + y < rows):  # i=0 y=0/1; i=rows-1 y=-1/0; i=rows y=-1
                        continue
                    for x in range(-1, 2):  # -1 0 1
                        if not (0 <= j + x < cols):  # j=0 x=0/1; j=cols-1 x=-1/0; j=cols x=-1
                            continue
                        if y == 0 and x == 0:
                            if dst[i, j] == 1:
                                a0 = -1
                                a1 = 0
                            else:
                                a0 = 1
                                a1 = 0
                        else:
                            if dst[i + y, j + x] != dst[i, j]:
                                if dst[i, j] == 1:
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0
                        # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
                        # 源码EPS
                        EPS = (a0 * a0 + a1 * a1) * CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                              2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y

                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP[y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1

        # 当一次迭代中没有修改，说明已经收敛，结束迭代
        if CountB == 0:
            break
        # break
        CountOpp += 1
        print("Opp = ", CountOpp, "  B = ", CountB)

    return dst


# dbs函数
# img: 原图像
# d: 滤波器参数
def dbs(img, d):
    fs = 7
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)

    CPP = np.zeros((13, 13))
    HalfCPPSize = 6
    CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))

    im = np.array(img, dtype=np.float64)
    print(im)

    rows, cols = im.shape

    dst = np.random.rand(rows, cols) > 0.5
    dst = np.where(dst, 1.0, 0.0)

    imr = im / 255.0
    Err = dst - imr

    CEP = signal.correlate2d(Err, CPP, mode='full')
    ESP_MIN = 0

    CountOpp = 0

    while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
        CountB = 0
        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):  # -1 0 1
                    if not (0 <= i + y < rows):  # i=0 y=0/1; i=rows-1 y=-1/0; i=rows y=-1
                        continue
                    for x in range(-1, 2):  # -1 0 1
                        if not (0 <= j + x < cols):  # j=0 x=0/1; j=cols-1 x=-1/0; j=cols x=-1
                            continue
                        if y == 0 and x == 0:
                            if dst[i, j] == 1:
                                a0 = -1
                                a1 = 0
                            else:
                                a0 = 1
                                a1 = 0
                        else:
                            if dst[i + y, j + x] != dst[i, j]:
                                if dst[i, j] == 1:
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0
                        # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
                        # 源码EPS
                        EPS = (a0 * a0 + a1 * a1) * CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                              2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y

                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP[y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1

        # 当一次迭代中没有修改，说明已经收敛，结束迭代
        if CountB == 0:
            break
        # break
        CountOpp += 1
        print("Opp = ", CountOpp, "  B = ", CountB)

    return dst

# clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
def clu_dbs(img, d, d1):
    fs = 7
    # d = (fs - 1) / 6
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

    # 初始化更新滤波器的参数
    # d1 = 1.2
    GF1 = np.zeros((fs, fs))

    for k in range(-gaulen, gaulen + 1):
        for l in range(-gaulen, gaulen + 1):
            c = (k ** 2 + l ** 2) / (2 * d ** 2)
            c1 = (k ** 2 + l ** 2) / (2 * d1 ** 2)
            GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
            GF1[k + gaulen, l + gaulen] = np.exp(-c1) / (2 * np.pi * d1 ** 2)

    CPP = np.zeros((13, 13))
    CPP1 = np.zeros((13, 13))
    HalfCPPSize = 6
    CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))
    CPP1 = CPP1 + (signal.convolve2d(GF1, GF1, mode='full', boundary='fill', fillvalue=0))

    # img = cv2.imread("resource/ctrf_sq1.jpg", cv2.IMREAD_GRAYSCALE)

    im = np.array(img, dtype=np.float64)

    rows, cols = im.shape

    dst = np.random.rand(rows, cols) > 0.5
    dst = np.where(dst, 1.0, 0.0)

    imr = im / 255.0
    Err = dst - imr

    CEP = signal.correlate2d(Err, CPP, mode='full')
    ESP_MIN = 0

    CountOpp = 0

    while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
        CountB = 0
        for i in range(rows):
            for j in range(cols):
                a0c = 0
                a1c = 0
                Cpx = 0
                Cpy = 0
                EPS_MIN = 0

                for y in range(-1, 2):
                    if not (0 <= i + y < rows):
                        continue
                    for x in range(-1, 2):
                        if not (0 <= j + x < cols):
                            continue
                        if y == 0 and x == 0:
                            if dst[i, j] == 1:
                                a0 = -1
                                a1 = 0
                            else:
                                a0 = 1
                                a1 = 0
                        else:
                            if dst[i + y, j + x] != dst[i, j]:
                                if dst[i, j] == 1:
                                    a0 = -1
                                    a1 = -a0
                                else:
                                    a0 = 1
                                    a1 = -a0
                            else:
                                a0 = 0
                                a1 = 0
                        # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
                        EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                              2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            a1c = a1
                            Cpx = x
                            Cpy = y

                if EPS_MIN < 0:
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP1[
                                y + HalfCPPSize, x + HalfCPPSize]
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP1[
                                y + HalfCPPSize, x + HalfCPPSize]
                    dst[i, j] += a0c
                    dst[i + Cpy, j + Cpx] += a1c
                    CountB += 1

        # 当一次迭代中没有修改，说明已经收敛，结束迭代
        if CountB == 0:
            break

        CountOpp += 1
        print("Opp = ", CountOpp, "  B = ", CountB)

    return dst

# mp_clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
# p: pass(迭代次数)
def mp_clu_dbs(img, dst, d, d1, p):
    # fs = 7
    # gaulen = int((fs - 1) / 2)
    # GF = np.zeros((fs, fs))
    #
    # # 初始化更新滤波器的参数
    # GF1 = np.zeros((fs, fs))
    #
    # for k in range(-gaulen, gaulen + 1):
    #     for l in range(-gaulen, gaulen + 1):
    #         c = (k ** 2 + l ** 2) / (2 * d ** 2)
    #         c1 = (k ** 2 + l ** 2) / (2 * d1 ** 2)
    #         GF[k + gaulen, l + gaulen] = np.exp(-c) / (2 * np.pi * d ** 2)
    #         GF1[k + gaulen, l + gaulen] = np.exp(-c1) / (2 * np.pi * d1 ** 2)
    #
    # CPP = np.zeros((13, 13))
    # CPP1 = np.zeros((13, 13))
    # HalfCPPSize = 6
    # CPP = CPP + (signal.convolve2d(GF, GF, mode='full', boundary='fill', fillvalue=0))
    # CPP1 = CPP1 + (signal.convolve2d(GF1, GF1, mode='full', boundary='fill', fillvalue=0))
    #
    # im = np.array(img, dtype=np.float64)
    #
    # rows, cols = im.shape
    #
    # imr = im / 255.0
    # Err = dst - imr
    #
    # CEP = signal.correlate2d(Err, CPP, mode='full')
    # ESP_MIN = 0
    #
    # CountOpp = 0
    #
    # while True:  # 这是用来循环迭代的参数，当一次迭代中修改像素的数量CountB=0，说明已经收敛
    #     CountB = 0
    #     for i in range(rows):
    #         for j in range(cols):
    #             a0c = 0
    #             a1c = 0
    #             Cpx = 0
    #             Cpy = 0
    #             EPS_MIN = 0
    #
    #             for y in range(-1, 2):
    #                 if not (0 <= i + y < rows):
    #                     continue
    #                 for x in range(-1, 2):
    #                     if not (0 <= j + x < cols):
    #                         continue
    #                     if y == 0 and x == 0:
    #                         if dst[i, j] == 1:
    #                             a0 = -1
    #                             a1 = 0
    #                         else:
    #                             a0 = 1
    #                             a1 = 0
    #                     else:
    #                         if dst[i + y, j + x] != dst[i, j]:
    #                             if dst[i, j] == 1:
    #                                 a0 = -1
    #                                 a1 = -a0
    #                             else:
    #                                 a0 = 1
    #                                 a1 = -a0
    #                         else:
    #                             a0 = 0
    #                             a1 = 0
    #                     # 以上代码在尝试进行翻转/交换操作，并计算操作后的ESP变化情况
    #                     EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
    #                           2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
    #                           2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
    #                           2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
    #                     if EPS_MIN > EPS:
    #                         EPS_MIN = EPS
    #                         a0c = a0
    #                         a1c = a1
    #                         Cpx = x
    #                         Cpy = y
    #
    #             if EPS_MIN < 0:
    #                 for y in range(-HalfCPPSize, HalfCPPSize + 1):
    #                     for x in range(-HalfCPPSize, HalfCPPSize + 1):
    #                         CEP[i + y + HalfCPPSize, j + x + HalfCPPSize] += a0c * CPP1[
    #                             y + HalfCPPSize, x + HalfCPPSize]
    #                 for y in range(-HalfCPPSize, HalfCPPSize + 1):
    #                     for x in range(-HalfCPPSize, HalfCPPSize + 1):
    #                         CEP[i + y + Cpy + HalfCPPSize, j + x + Cpx + HalfCPPSize] += a1c * CPP1[
    #                             y + HalfCPPSize, x + HalfCPPSize]
    #                 dst[i, j] += a0c
    #                 dst[i + Cpy, j + Cpx] += a1c
    #                 CountB += 1
    #
    #     # 当一次迭代中没有修改，说明已经收敛，结束迭代
    #     if CountB == 0:
    #         break
    #
    #     CountOpp += 1
    #     print("Opp = ", CountOpp, "  B = ", CountB)
    for i in range(p):
        dst = clu_dbs(img, d, d1)
        img = dst

    return dst

# ms_mp_clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
# s: stage(阶段数)
# p: pass(迭代次数)
def ms_mp_clu_dbs(img, dst, d, d1, s, p):
    # img1 = 255 - (255 - img) // 5
    # img2 = 255 - (255 - img) // 5 * 2
    # img3 = 255 - (255 - img) // 5 * 3
    # img4 = 255 - (255 - img) // 5 * 4
    # img5 = 255 - (255 - img) // 5 * 5
    #
    # for i in range(5):
    #     dst = mp_clu_dbs(img1, dst, d, d1)
    # for i in range(5):
    #     dst = mp_clu_dbs(img2, dst, d, d1)
    # for i in range(5):
    #     dst = mp_clu_dbs(img3, dst, d, d1)
    # for i in range(5):
    #     dst = mp_clu_dbs(img4, dst, d, d1)
    # for i in range(5):
    #     dst = mp_clu_dbs(img5, dst, d, d1)

    for i in range(s):
        imgi = 255 - (255 - img) // s * (i + 1)
        dst = mp_clu_dbs(imgi, dst, d, d1, p)
    return dst



if __name__ == '__main__':
    img = cv2.imread("resource/ctrf_sq1.jpg", cv2.IMREAD_GRAYSCALE)

    dst = seed(img, 1.3, 7.57/255)
    dst = ms_mp_clu_dbs(img, dst, 1.3, 1.7, 5, 10)
    cv2.imshow("image", dst)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
