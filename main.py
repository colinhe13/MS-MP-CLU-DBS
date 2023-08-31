import numpy as np
import cv2
import scipy.signal as signal


# 产生种子半色调图像的函数
# img: 原图像
# d: 滤波器参数
# g: 种子半色调的灰度级数 0-255/255
def get_seed(img, d, g):
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

    # imr = np.full((rows, cols), 1 - g, dtype=np.float64)
    imr = np.full((rows, cols), g, dtype=np.float64)

    dst = np.random.rand(rows, cols) > g
    # dst = np.where(dst, 1.0, 0.0)
    dst = np.where(dst, 0.0, 1.0)

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
                            continue
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


# 从种子半色调生成初始化半色调
# 每个8*8区域内计算原始图像平均灰度值，然后通过在种子相同位置的随机翻转匹配灰度
def initialHalftone(img, dst):
    # cv2.imshow("before initial", dst)
    # cv2.waitKey(0)
    im = np.array(img, dtype=np.float64)
    rows, cols = im.shape
    size = 8
    rowsi = rows // size
    colsi = cols // size

    for i in range(rowsi):
        for j in range(colsi):
            countg = 0
            gray_value = 0.0
            for y in range(size):
                for x in range(size):
                    gray_value += im[i * size + y, j * size + x] / 255

            # gray_value = 1 - (gray_value / size ** 2)
            gray_value = gray_value / size ** 2

            # 0-->white  1-->black
            bnum = round(size ** 2 * gray_value)
            for y in range(size):
                for x in range(size):
                    # if dst[i * size + y, j * size + x] == 0:
                    #     dst[i * size + y, j * size + x] = 1
                    if dst[i * size + y, j * size + x] == 1:
                        dst[i * size + y, j * size + x] = 0
                    # if dst[i * size + y, j * size + x] == 1:
                    #     bnum -= 1
            while bnum > 0:
                x = np.random.randint(0, size)
                y = np.random.randint(0, size)
                if dst[i * size + y, j * size + x] == 0:
                    dst[i * size + y, j * size + x] = 1
                    bnum -= 1
                countg += 1
            # while bnum < 0:
            #     x = np.random.randint(0, size)
            #     y = np.random.randint(0, size)
            #     if dst[i * size + y, j * size + x] == 1:
            #         dst[i * size + y, j * size + x] = 0
            #         bnum += 1
            #     countg += 1
            print("countg = ", countg)
    # cv2.imshow("after initial", dst)
    # cv2.waitKey(0)
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
def clu_dbs(img, dst, d, d1):
    fs = 7
    gaulen = int((fs - 1) / 2)
    GF = np.zeros((fs, fs))

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

    im = np.array(img, dtype=np.float64)

    rows, cols = im.shape

    # dst = np.random.rand(rows, cols) > 0.5
    # dst = np.where(dst, 1.0, 0.0)

    # gray_value = 0.0
    # for y in range(rows):
    #     for x in range(cols):
    #         gray_value += dst[y, x]
    # gray_value = gray_value / (rows * cols)

    imr = im / 255.0

    # Err0 = dst - np.full((rows, cols), gray_value, dtype=np.float64)
    Err0 = dst - imr
    D_CEP0 = signal.correlate2d(Err0, CPP-CPP1, mode='full')

    dst = initialHalftone(im, dst)


    Err = dst - imr
    # Err = seed - imr

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
                                # a0 = 1
                                a1 = 0
                            else:
                                a0 = 1
                                # a0 = 0
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
                        # EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
                        #       2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
                        #       2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize] + \
                        #       2 * a1 * CEP[i + y + HalfCPPSize, j + x + HalfCPPSize]
                        EPS = (a0 * a0 + a1 * a1) * CPP1[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * a1 * CPP1[HalfCPPSize + y, HalfCPPSize + x] + \
                              2 * a0 * CEP[HalfCPPSize + i, HalfCPPSize + j] - \
                              2 * a0 * D_CEP0[HalfCPPSize + i, HalfCPPSize + j] + \
                              2 * a0 * a1 * a0 * CEP[HalfCPPSize + i + y, HalfCPPSize + j + x] - \
                              2 * a0 * a1 * a0 * D_CEP0[HalfCPPSize + i + y, HalfCPPSize + j + x]

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

    cv2.imshow("after cludbs", dst)
    cv2.waitKey(0)

    return dst


# mp_clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
# p: pass(迭代次数)
def mp_clu_dbs(img, dst, d, d1, p):
    CountInit = 0
    for i in range(p):
        if CountInit == 0:
            CountInit += 0.25
        elif CountInit == 0.25:
            CountInit += 0.25
        else:
            CountInit = 1
        imgi = img * CountInit
        # dst = initialHalftone(imgi, dst)
        dst = clu_dbs(img, dst, d, d1)

    return dst


# ms_mp_clu_dbs函数
# img: 原图像
# d: 初始滤波器参数
# d1: 更新滤波器参数
# s: stage(阶段数)
# p: pass(迭代次数)
def ms_mp_clu_dbs(img, dst, d, d1, s, p):
    for i in range(s):
        # imgi = 255 - (255 - img) / s * (i + 1)
        imgi = img / s * (i + 1)

        # dst = initialHalftone(imgi, dst)
        dst = mp_clu_dbs(imgi, dst, d, d1, p)
        save_filepath = "output/init12/msmpcludbs_seed7_d13_17_s5_p10_gray127_0w1b_stage" + str(i) + ".npy"
        # np.save(save_filepath, dst)

    return dst


# 生成ms_mp_clu_dbs网屏的函数
# height: 网屏高度
# width: 网屏宽度
def screen_msmpcludbs(height, width, g, d, d1, s, p):
    screen = np.full((height, width), 0, dtype=np.uint8)

    # 首先生成灰度级127/255的ms_mp_clu_dbs半色调图像并存入screen数组
    mid_img = np.full((height, width), 127, dtype=np.uint8)
    dst = get_seed(screen, d, g)
    dst = ms_mp_clu_dbs(mid_img, dst, d, d1, s, p)
    # np.save("output/gray_values_127.npy", dst)
    for i in range(height):
        for j in range(width):
            if dst[i, j] == 0:
                screen[i, j] = 127
            else:
                screen[i, j] = 255

    # 调整灰度级127/255图像中黑色像素的数量
    screen = fill_screen(screen, 127, 127, d)

    # 生成灰度级126/255到0/255的ms_mp_clu_dbs半色调图像
    for k in range(126, -1, -1):
        screen = fill_screen(screen, k + 1, k, d)

    # 生成灰度级128/255到255/255的ms_mp_clu_dbs半色调图像
    for k in range(128, 256):
        screen = fill_screen(screen, k - 1, k, d)

    return screen


# 根据输入的灰度级数在网屏矩阵中填入对应级数的函数
# screen: 网屏矩阵
# cur_values: 当前灰度级数
# target_values: 目标灰度级数
# d: 滤波器参数
def fill_screen(screen, cur_values, target_values, d):
    height, width = screen.shape
    CPP = cal_cpp(d)
    HalfCPPSize = 6

    im = np.full((height, width), target_values, dtype=np.float64)
    imr = im / 255.0

    # 情况1 目标值=当前值：重新检查网屏中灰度值是否符合要求
    if target_values == cur_values:
        dst = screen == cur_values
        dst = np.where(dst, 0.0, 1.0)

        Err = dst - imr
        CEP = signal.correlate2d(Err, CPP, mode='full')

        num_black = np.count_nonzero(dst == 0.0)
        num_target = round(target_values / 255 * height * width)

        while num_black != num_target:
            # 如果黑色像素数量过多
            if num_black > num_target:
                # 找到使误差降低最多或增加最少的黑色像素0，并将其修改为白色1
                EPS_MIN, EPS = 0, 0
                a0c, Cpx, Cpy = 0, 0, 0
                flag = True
                # 先找EPS降低
                if flag:
                    flag = False
                    EPS_MIN = 0
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 0:
                                a0 = 1
                                EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                                    flag = True
                                    print("working on i = ", i, "  j = ", j)
                                    print("max smaller EPS = ", EPS)
                    dst[Cpy, Cpx] += a0c
                    num_black -= 1
                    screen[Cpy, Cpx] = 255
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        break
                # 如果数量还是不够，再找EPS增加
                if not flag:
                    EPS_MIN = 100000
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 0:
                                a0 = 1
                                EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                                    print("working on i = ", i, "  j = ", j)
                                    print("min larger EPS = ", EPS)
                    dst[Cpy, Cpx] += a0c
                    num_black -= 1
                    screen[Cpy, Cpx] = 255
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        break
            else:
                # 找到使误差降低最多或增加最少的白色像素1，并将其修改为黑色0
                EPS_MIN, EPS = 0, 0
                a0c, Cpx, Cpy = 0, 0, 0
                flag = True
                # 先找EPS降低
                if flag:
                    flag = False
                    EPS_MIN = 0
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 1:
                                a0 = -1
                                EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                                    flag = True
                                    print("working on i = ", i, "  j = ", j)
                                    print("max smaller EPS = ", EPS)
                    dst[Cpy, Cpx] += a0c
                    num_black += 1
                    screen[Cpy, Cpx] = target_values
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        break
                # 如果数量还是不够，再找EPS增加
                if not flag:
                    EPS_MIN = 100000
                    for i in range(height):
                        for j in range(width):
                            if dst[i, j] == 1:
                                a0 = -1
                                EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                                      2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                                if EPS_MIN > EPS:
                                    EPS_MIN = EPS
                                    a0c = a0
                                    Cpx = j
                                    Cpy = i
                                    print("working on i = ", i, "  j = ", j)
                                    print("min larger EPS = ", EPS)
                    dst[Cpy, Cpx] += a0c
                    num_black += 1
                    screen[Cpy, Cpx] = target_values
                    for y in range(-HalfCPPSize, HalfCPPSize + 1):
                        for x in range(-HalfCPPSize, HalfCPPSize + 1):
                            CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                                y + HalfCPPSize, x + HalfCPPSize]
                    if num_black == num_target:
                        break

        return screen

    # 情况2 目标值<当前值：找到并修改使误差降低最多或增加最少的黑色像素
    elif target_values < cur_values:
        # dst初始化为screen中所有值==cur_values的像素
        dst = screen == cur_values
        dst = np.where(dst, 0.0, 1.0)

        Err = dst - imr
        CEP = signal.correlate2d(Err, CPP, mode='full')

        num_black = np.count_nonzero(dst == 0.0)
        num_target = round(target_values / 255 * height * width)

        while num_black != num_target:
            # 找到使误差降低最多或增加最少的黑色像素0，并将其修改为白色1
            EPS_MIN, EPS = 0, 0
            a0c, Cpx, Cpy = 0, 0, 0
            flag = True
            # 先找EPS降低
            while flag:
                flag = False
                EPS_MIN = 0
                for i in range(height):
                    for j in range(width):
                        if dst[i, j] == 0:
                            a0 = 1
                            EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                                  2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                            if EPS_MIN > EPS:
                                EPS_MIN = EPS
                                a0c = a0
                                Cpx = j
                                Cpy = i
                                flag = True
                dst[Cpy, Cpx] += a0c
                screen[Cpy, Cpx] = target_values
                num_black -= 1
                for y in range(-HalfCPPSize, HalfCPPSize + 1):
                    for x in range(-HalfCPPSize, HalfCPPSize + 1):
                        CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[y + HalfCPPSize, x + HalfCPPSize]
                if num_black == num_target:
                    flag = False
                    break
            # 如果数量还是不够，再找EPS增加
            while flag:
                flag = False
                EPS_MIN = 100000
                for i in range(height):
                    for j in range(width):
                        if dst[i, j] == 0:
                            a0 = 1
                            EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                                  2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                            if EPS_MIN > EPS:
                                EPS_MIN = EPS
                                a0c = a0
                                Cpx = j
                                Cpy = i
                                flag = True
                dst[Cpy, Cpx] += a0c
                screen[Cpy, Cpx] = target_values
                num_black -= 1
                for y in range(-HalfCPPSize, HalfCPPSize + 1):
                    for x in range(-HalfCPPSize, HalfCPPSize + 1):
                        CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                            y + HalfCPPSize, x + HalfCPPSize]
                if num_black == num_target:
                    break
        return screen

    # 情况3 目标值>当前值：找到并修改使误差降低最多或增加最少的白色像素
    else:
        # dst初始化为screen中所有值<=cur_values的像素
        dst = screen <= cur_values
        dst = np.where(dst, 0.0, 1.0)

        Err = dst - imr
        CEP = signal.correlate2d(Err, CPP, mode='full')

        num_black = np.count_nonzero(dst == 0.0)
        num_target = round(target_values / 255 * height * width)

        # 找到使误差降低最多或增加最少的白色像素1，并将其修改为黑色0
        EPS_MIN, EPS = 0, 0
        a0c, Cpx, Cpy = 0, 0, 0
        flag = True
        # 先找EPS降低
        while flag:
            flag = False
            EPS_MIN = 0
            for i in range(height):
                for j in range(width):
                    if dst[i, j] == 1:
                        a0 = -1
                        EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            Cpx = j
                            Cpy = i
                            flag = True
            dst[Cpy, Cpx] += a0c
            num_black += 1
            screen[Cpy, Cpx] = target_values
            for y in range(-HalfCPPSize, HalfCPPSize + 1):
                for x in range(-HalfCPPSize, HalfCPPSize + 1):
                    CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                        y + HalfCPPSize, x + HalfCPPSize]
            if num_black == num_target:
                break
        # 如果数量还是不够，再找EPS增加
        while flag:
            flag = False
            EPS_MIN = 100000
            for i in range(height):
                for j in range(width):
                    if dst[i, j] == 1:
                        a0 = -1
                        EPS = CPP[HalfCPPSize, HalfCPPSize] + \
                              2 * a0 * CEP[i + HalfCPPSize, j + HalfCPPSize]
                        if EPS_MIN > EPS:
                            EPS_MIN = EPS
                            a0c = a0
                            Cpx = j
                            Cpy = i
                            flag = True
            dst[Cpy, Cpx] += a0c
            num_black += 1
            screen[Cpy, Cpx] = target_values
            for y in range(-HalfCPPSize, HalfCPPSize + 1):
                for x in range(-HalfCPPSize, HalfCPPSize + 1):
                    CEP[Cpy + y + HalfCPPSize, Cpx + x + HalfCPPSize] += a0c * CPP[
                        y + HalfCPPSize, x + HalfCPPSize]
            if num_black == num_target:
                break

        return screen


# 计算CPP的函数(CPP在整个计算过程中不改变)
def cal_cpp(d):
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

    return CPP


# 特定翻转操作(找到并修改使误差降低最多或增加最少的黑色/白色像素)
def particular_toggle():
    return


if __name__ == '__main__':
    # img = cv2.imread("resource/ctrf_sq1.jpg", cv2.IMREAD_GRAYSCALE)
    # img = np.full((256, 256), 127, dtype=np.uint8)
    img = np.full((128, 128), 127, dtype=np.uint8)

    # dst = get_seed(img, 1.3, 7.57 / 255)
    dst = get_seed(img, 1.3, 7.57 / 255)
    # dst = get_seed(img, 1.3, 128/255)

    # np.save("output/seed_7_size_256_256.npy", dst)
    # dst = np.load("output/seed.npy")
    # dst = np.load("output/seed7_d13_size256_0w1b.npy")
    # dst = np.load("output/seed_7_size_256_256_cut.npy")
    # dst = np.load("output/seed_128_size_256_256.npy")
    # dst = np.load("output/msmpcludbs_seed7_init8_d13_17_s10_p5_gray127.npy")

    # dst = np.random.rand(256, 256) > 0.5
    # dst = np.where(dst, 1.0, 0.0)

    # cv2.imshow("seed", dst)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # dst = clu_dbs(img, dst, 1.3, 1.7)
    dst = mp_clu_dbs(img, dst, 1.3, 1.7, 10)
    # dst = ms_mp_clu_dbs(img, dst, 1.3, 1.7, 5, 10)
    # dst = np.where(dst == 0, 127, 255)
    # dst = fill_screen(dst, 127, 127, 1.3)

    # np.save("output/seed7_d13_size256_0w1b.npy", dst)
    # np.save("output/cludbs_seed7_d13_17_gray127.npy", dst)
    # np.save("output/mpcludbs_seed128_d13_17_p10_gray127.npy", dst)
    # np.save("output/msmpcludbs_seed7_init3_d13_17_s5_p10_gray127.npy", dst)
    # np.save("output/msmpcludbs_seed0_d13_17_s5_p10_gray127.npy", dst)
    # save_filepath = "output/msmpcludbs_seed7_init9_d13_17_s5_p10_gray127_0w1b.npy"
    save_filepath = "output/init14/mpcludbs_seed7_d13_17_pd10_gray127.npy"
    np.save(save_filepath, dst)

    print("##################################################################" + "\n"
            "       Program Completed       " + "\n"
            "       Result has Saved to " + save_filepath + "\n"
            "##################################################################")

    # dst = np.where(dst == 127, 0, 1)
    # cv2.imwrite("output/cludbs_seed7_d1_12_ctrf.png", dst)
    cv2.imshow("image", dst)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # height = 256
    # width = 256
    # gray_values = np.full((height, width), 127, dtype=np.uint8)
    #
    # dst = seed(gray_values, 1.3, 7.57 / 255)
    # dst = clu_dbs(gray_values, dst, 1.3, 1.7)

    # np.save("output/gray_values.npy", dst)

    # cv2.imwrite("output/gray_values.jpg", dst)
    # cv2.imshow("image", dst)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()
