import xlrd
import numpy as np

def xy_to_uv(x, y):
    """
        将CIE 1931色品坐标(x,y)转换为CIE 1960色品坐标(u,v)
    """
    u = 4 * x / (-2 * x + 12 * y + 3)
    v = 6 * y / (-2 * x + 12 * y + 3)
    return u, v


def duv_formula_method(xt, yt):
    """
        计算Duv
    """
    # 读取黑体曲线色品坐标数据
    data = xlrd.open_workbook('data/black_body_locus.xls')
    table = data.sheets()[0]
    m = table.nrows
    n = table.ncols
    # 将读取的数据赋给数组black_locus
    black_locus = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            black_locus[i, j] = table.cell(i, j).value
    # 将CIE 1931色品坐标转换为CIE 1960色品坐标
    xp = black_locus[:, 1]
    yp = black_locus[:, 2]
    # 转换为CIE 1960色品坐标
    [up, vp] = xy_to_uv(xp, yp)
    # 将待测光源色品坐标作同样的转换
    [ut, vt] = xy_to_uv(xt, yt)
    # 用公式(1-221)求色品坐标之间的距离
    delta_c = ((ut-up)**2+(vt-vp)**2)**0.5
    # 返回距离最小值的索引
    n = np.argmin(delta_c)
    # 对Duv的符号进行判定
    if vt-vp[n] >= 0:
        sign_flag = 1
    else:
        sign_flag = -1
    # 计算Duv
    duv = sign_flag*delta_c[n]
    # 返回Duv
    return duv

def main():
    # 设置CIE 1931色品坐标
    xt = 0.3840445003795555
    yt = 0.3767800565662821
    duv = duv_formula_method(xt, yt)
    print('Duv =', round(duv, 6))

if __name__ == '__main__':
    main()