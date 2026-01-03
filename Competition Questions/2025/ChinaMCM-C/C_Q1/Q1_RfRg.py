import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from scipy.integrate import simpson
from scipy.optimize import brentq

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 相关色温计算 - Chebyshev法
def calculate_cct_chebyshev(x, y):
    """使用黑体轨迹的Chebyshev法计算相关色温"""
    # 转换为CIE 1960 (u, v)
    denominator = -2 * x + 12 * y + 3
    u_c = 4 * x / denominator
    v_c = 6 * y / denominator
    
    # 黑体轨迹函数
    def u_bar(T):
        numerator = 0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T**2
        denominator = 1 + 8.42420235e-4 * T + 7.08145163e-7 * T**2
        return numerator / denominator
    
    def v_bar(T):
        numerator = 0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T**2
        denominator = 1 - 2.89741816e-5 * T + 1.61456053e-7 * T**2
        return numerator / denominator
    
    # 数值微分(中心差分)
    def derivative(func, T, h=0.1):
        return (func(T + h) - func(T - h)) / (2 * h)
    
    # 目标函数
    def objective(T):
        u_t = u_bar(T)
        v_t = v_bar(T)
        du_dT = derivative(u_bar, T)
        dv_dT = derivative(v_bar, T)
        return du_dT * (u_t - u_c) + dv_dT * (v_t - v_c)
    
    # 求解相关色温
    try:
        T_c = brentq(objective, 1000, 15000, xtol=1e-5)
        return T_c
    except ValueError:
        # 分段搜索确保找到解
        T_values = np.linspace(1000, 15000, 100)
        f_values = [objective(T) for T in T_values]
        for i in range(len(T_values) - 1):
            if f_values[i] * f_values[i+1] <= 0:
                T_c = brentq(objective, T_values[i], T_values[i+1], xtol=1e-5)
                return T_c
        raise ValueError("无法计算Chebyshev法相关色温")


# 相关色温计算 - McCamy近似公式法
def calculate_cct_mccamy(x, y):
    """使用McCamy近似公式法计算相关色温"""
    n = (x - 0.3320) / (y - 0.1858)
    T = -437 * n**3 + 3601 * n**2 - 6861 * n + 5514.31
    return T


# 数据加载函数
def load_data():
    # 读取待测光源SPD
    spd_data = pd.read_csv('data/Problem_1.csv')
    spd_wavelengths = spd_data['wavelength'].values
    spd_values = spd_data['SPD'].values
    
    # 读取10°视角CIE三刺激值
    cie10_data = pd.read_csv('data/CIE_xyz_1964_10deg.csv')
    cie10_wavelengths = cie10_data['wavelength'].values
    cie10_x = cie10_data['x'].fillna(0).values
    cie10_y = cie10_data['y'].fillna(0).values
    cie10_z = cie10_data['z'].fillna(0).values
    
    # 读取2°视角CIE三刺激值（用于相关色温计算）
    cie2_data = pd.read_csv('data/CIE_xyz_1931_2deg.csv')
    cie2_wavelengths = cie2_data['wavelength'].values
    cie2_x = cie2_data['x'].fillna(0).values
    cie2_y = cie2_data['y'].fillna(0).values
    cie2_z = cie2_data['z'].fillna(0).values
    
    # 读取99个颜色样品反射率
    color_samples = pd.read_csv('data/99_color_samples.csv')
    sample_wavelengths = color_samples['wavelength'].values
    samples = color_samples.filter(regex='sample').values.T  # 99x401矩阵
    
    # 读取重组日光矢量数据
    daylight_data = pd.read_csv('data/common_daylight_1nm.csv')
    daylight_wavelengths = daylight_data['wavelength'].values
    daylight_v1 = daylight_data['V1'].values  # 基础矢量1
    daylight_v2 = daylight_data['V2'].values  # 基础矢量2
    daylight_v3 = daylight_data['V3'].values  # 基础矢量3
    
    return {
        'spd': (spd_wavelengths, spd_values),
        'cie10': (cie10_wavelengths, cie10_x, cie10_y, cie10_z),
        'cie2': (cie2_wavelengths, cie2_x, cie2_y, cie2_z),
        'samples': (sample_wavelengths, samples),
        'daylight': (daylight_wavelengths, daylight_v1, daylight_v2, daylight_v3)
    }

# 黑体光谱生成
def generate_blackbody_spectrum(tc, wavelengths):
    h = 6.62607015e-34
    c = 299792458
    k = 1.380649e-23
    a = 2 * np.pi * h * c **2
    b = h * c / k
    
    p_lambda = []
    for wl in wavelengths:
        lambda_m = wl * 1e-9  # 转换为米
        exponent = b / (lambda_m * tc)
        if exponent > 700:  # 避免指数溢出
            p = 0.0
        else:
            p = a * (lambda_m** -5) / (np.exp(exponent) - 1)
        p_lambda.append(p)
    
    # 归一化
    max_p = max(p_lambda) if p_lambda else 1.0
    return np.array(p_lambda) / max_p

# 重组日光光谱生成
def composed_daylight(tc, wavelengths, daylight_v1, daylight_v2, daylight_v3):
    """根据重组日光矢量计算光谱"""
    # 计算重组日光的色度坐标xD, yD
    if tc > 25000:
        xd = 0.250
    elif tc > 7000:
        xd = -2.0064e9 / (tc ** 3) + 1.9018e6 / (tc ** 2) + 0.24748e3 / tc + 0.237040
    elif tc >= 4000:
        xd = -4.6070e9 / (tc ** 3) + 2.9678e6 / (tc ** 2) + 0.09911e3 / tc + 0.244063
    else:
        xd = 0.380  # 低于4000K时的默认值
    yd = -3.000 * (xd ** 2) + 2.870 * xd - 0.275  # yD与xD的关系公式
    
    # 计算矢量因子M1和M2
    denominator = 0.0241 + 0.2562 * xd - 0.7341 * yd
    if abs(denominator) < 1e-9:  # 避免分母为0
        m1, m2 = 0.0, 0.0
    else:
        m1 = (-1.3515 - 1.7703 * xd + 5.9114 * yd) / denominator
        m2 = (0.0300 - 31.4424 * xd + 30.0717 * yd) / denominator
    
    # 生成重组日光光谱
    spectrum = daylight_v1 + m1 * daylight_v2 + m2 * daylight_v3
    
    # 确保光谱非负
    spectrum[spectrum < 0] = 0.0
    return spectrum

# 混合光谱生成
def mix_blackbody_daylight(tc, wavelengths, cie2_x, cie2_y, cie2_z, daylight_v1, daylight_v2, daylight_v3):
    # 生成黑体和重组日光光谱
    srp = generate_blackbody_spectrum(tc, wavelengths)
    srd = composed_daylight(tc, wavelengths, daylight_v1, daylight_v2, daylight_v3)
    
    # 按2°视角明度函数归一化
    sum_y_srp = simpson(srp * cie2_y, wavelengths)
    srp = 100 * srp / sum_y_srp if sum_y_srp != 0 else srp
    sum_y_srd = simpson(srd * cie2_y, wavelengths)
    srd = 100 * srd / sum_y_srd if sum_y_srd != 0 else srd
    
    # 线性混合
    tb, te = 4000, 5000
    return (tc - tb)/(te - tb) * srd + (te - tc)/(te - tb) * srp

# 改进的相关色温计算函数
def correlated_color_temperature(spectrum, wavelengths, cie2_x, cie2_y, cie2_z, method='chebyshev'):
    """
    计算相关色温，支持两种方法
    
    参数:
        spectrum: 光源光谱
        wavelengths: 波长数组
        cie2_x, cie2_y, cie2_z: 2°视角三刺激值
        method: 计算方法，'chebyshev'或'mccamy'
    
    返回:
        相关色温值(K)
    """
    # 计算三刺激值
    sum_x = simpson(spectrum * cie2_x, wavelengths)
    sum_y = simpson(spectrum * cie2_y, wavelengths)
    sum_z = simpson(spectrum * cie2_z, wavelengths)
    
    # 计算色品坐标
    sum_xyz = sum_x + sum_y + sum_z
    if sum_xyz == 0:
        raise ValueError("无法计算三刺激值总和为零的光谱色温")
    
    x = sum_x / sum_xyz
    y = sum_y / sum_xyz
    
    # 选择计算方法
    if method == 'chebyshev':
        return calculate_cct_chebyshev(x, y)
    elif method == 'mccamy':
        return calculate_cct_mccamy(x, y)
    else:
        raise ValueError("不支持的计算方法，可选'chebyshev'或'mccamy'")

# 光源三刺激值计算
def light_source_stimulus(spectrum, wavelengths, cie_x, cie_y, cie_z):
    sum_y = simpson(spectrum * cie_y, wavelengths)
    k = 100 / sum_y if sum_y != 0 else 1.0
    x = k * simpson(spectrum * cie_x, wavelengths)
    z = k * simpson(spectrum * cie_z, wavelengths)
    return {'x_capital': x, 'y_capital': 100.0, 'z_capital': z, 'k': k}

# 色品坐标计算
def get_chromaticity(x, y, z):
    sum_xyz = x + y + z
    return (x/sum_xyz, y/sum_xyz) if sum_xyz != 0 else (0.0, 0.0)

# 样品三刺激值计算
def color_sample_stimulus(sample, spectrum, wavelengths, k, cie_x, cie_y, cie_z):
    x = k * simpson(sample * spectrum * cie_x, wavelengths)
    y = k * simpson(sample * spectrum * cie_y, wavelengths)
    z = k * simpson(sample * spectrum * cie_z, wavelengths)
    return x, y, z

# 色适应转换
def von_kries_adapt(r, g, b, rw, gw, bw):
    rc = 100 * r / rw if rw != 0 else 0
    gc = 100 * g / gw if gw != 0 else 0
    bc = 100 * b / bw if bw != 0 else 0
    return rc, gc, bc

# XYZ转RGB
def xyz_to_rgb(x, y, z):
    mcat02 = [
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834]
    ]
    return (
        mcat02[0][0]*x + mcat02[0][1]*y + mcat02[0][2]*z,
        mcat02[1][0]*x + mcat02[1][1]*y + mcat02[1][2]*z,
        mcat02[2][0]*x + mcat02[2][1]*y + mcat02[2][2]*z
    )

# RGB转XYZ
def rgb_to_xyz(r, g, b):
    m1cat02 = [
        [1.096124, -0.278869, 0.182745],
        [0.454369, 0.473533, 0.072098],
        [-0.009628, -0.005698, 1.015326]
    ]
    return (
        m1cat02[0][0]*r + m1cat02[0][1]*g + m1cat02[0][2]*b,
        m1cat02[1][0]*r + m1cat02[1][1]*g + m1cat02[1][2]*b,
        m1cat02[2][0]*r + m1cat02[2][1]*g + m1cat02[2][2]*b
    )

# Hunt-Pointer-Estevez变换
def hunt_pointer_estevez(x, y, z):
    hpe = [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0, 0, 1]
    ]
    return (
        hpe[0][0]*x + hpe[0][1]*y + hpe[0][2]*z,
        hpe[1][0]*x + hpe[1][1]*y + hpe[1][2]*z,
        hpe[2][0]*x + hpe[2][1]*y + hpe[2][2]*z
    )

# 亮度适应
def luminance_adaptation(rp, gp, bp):
    fl = 0.7937
    def adapt(v):
        val = (fl * v / 100) **0.42
        return 400 * val / (27.13 + val) + 0.1
    return adapt(rp), adapt(gp), adapt(bp)

# ab通道计算
def channel_ab(r, g, b):
    aw = r - 12/11 * g + 1/11 * b
    bw = 1/9 * (r + g - 2*b)
    aW = 1.0003 * (2*r + g + 1/20*b - 0.305)
    return aw, bw, aW

# 色调角分箱
def count_hue_angle(hue_angles):
    return [int(angle / 22.5) + 1 for angle in hue_angles]

# 多边形面积计算
def polygon_area(a, b):
    if len(a) != len(b):
        raise ValueError("数组长度必须相同")
    n = len(a)
    a = np.append(a, a[0])
    b = np.append(b, b[0])
    area = 0.0
    for i in range(n):
        area += a[i] * b[i+1] - a[i+1] * b[i]
    return 0.5 * abs(area)

# 主计算函数
def calculate_rf_rg_with_visual_data():
    data = load_data()
    spd_wavelengths, spd_values = data['spd']
    cie10_wl, cie10_x, cie10_y, cie10_z = data['cie10']
    cie2_wl, cie2_x, cie2_y, cie2_z = data['cie2']
    sample_wl, samples = data['samples']
    daylight_wl, daylight_v1, daylight_v2, daylight_v3 = data['daylight']
    
    # 波长一致性检查
    if not (np.array_equal(spd_wavelengths, cie10_wl) and 
            np.array_equal(spd_wavelengths, cie2_wl) and 
            np.array_equal(spd_wavelengths, sample_wl) and
            np.array_equal(spd_wavelengths, daylight_wl)):
        raise ValueError("所有数据的波长必须一致")
    
    # 待测光谱归一化
    max_spd = max(spd_values) if spd_values.size else 1.0
    testing_spectrum = spd_values / max_spd
    
    # 计算相关色温（使用Chebyshev法，更精确）
    tc_chebyshev = correlated_color_temperature(
        testing_spectrum, spd_wavelengths, cie2_x, cie2_y, cie2_z, method='chebyshev'
    )
    tc_mccamy = correlated_color_temperature(
        testing_spectrum, spd_wavelengths, cie2_x, cie2_y, cie2_z, method='mccamy'
    )
    print(f"Chebyshev法计算的相关色温: {tc_chebyshev:.0f} K")
    print(f"McCamy法计算的相关色温: {tc_mccamy:.0f} K")
    # 使用更精确的Chebyshev法结果
    tc = tc_chebyshev
    
    # 根据色温选择参照光谱
    if tc < 4000:
        reference_spectrum = generate_blackbody_spectrum(tc, spd_wavelengths)
    elif tc > 5000:
        reference_spectrum = composed_daylight(tc, spd_wavelengths, daylight_v1, daylight_v2, daylight_v3)
    else:
        reference_spectrum = mix_blackbody_daylight(
            tc, spd_wavelengths, cie2_x, cie2_y, cie2_z,
            daylight_v1, daylight_v2, daylight_v3
        )
    
    # 后续计算逻辑
    test_stim = light_source_stimulus(testing_spectrum, spd_wavelengths, cie10_x, cie10_y, cie10_z)
    ref_stim = light_source_stimulus(reference_spectrum, spd_wavelengths, cie10_x, cie10_y, cie10_z)
    
    test_x, test_y = get_chromaticity(test_stim['x_capital'], test_stim['y_capital'], test_stim['z_capital'])
    ref_x, ref_y = get_chromaticity(ref_stim['x_capital'], ref_stim['y_capital'], ref_stim['z_capital'])
    
    sample_count = 99
    test_samples = []
    ref_samples = []
    for i in range(sample_count):
        sample = samples[i] if i < len(samples) else np.zeros_like(spd_values)
        tx, ty, tz = color_sample_stimulus(sample, testing_spectrum, spd_wavelengths, test_stim['k'], cie10_x, cie10_y, cie10_z)
        rx, ry, rz = color_sample_stimulus(sample, reference_spectrum, spd_wavelengths, ref_stim['k'], cie10_x, cie10_y, cie10_z)
        test_samples.append((tx, ty, tz))
        ref_samples.append((rx, ry, rz))
    
    test_rgb = xyz_to_rgb(test_stim['x_capital'], test_stim['y_capital'], test_stim['z_capital'])
    ref_rgb = xyz_to_rgb(ref_stim['x_capital'], ref_stim['y_capital'], ref_stim['z_capital'])
    
    test_rc, test_gc, test_bc = von_kries_adapt(*test_rgb, *test_rgb)
    ref_rc, ref_gc, ref_bc = von_kries_adapt(*ref_rgb, *ref_rgb)
    
    test_xct, test_yct, test_zct = rgb_to_xyz(test_rc, test_gc, test_bc)
    ref_xcr, ref_ycr, ref_zcr = rgb_to_xyz(ref_rc, ref_gc, ref_bc)
    
    test_rp, test_gp, test_bp = hunt_pointer_estevez(test_xct, test_yct, test_zct)
    ref_rp, ref_gp, ref_bp = hunt_pointer_estevez(ref_xcr, ref_ycr, ref_zcr)
    
    test_ra, test_ga, test_ba = luminance_adaptation(test_rp, test_gp, test_bp)
    ref_ra, ref_ga, ref_ba = luminance_adaptation(ref_rp, ref_gp, ref_bp)
    
    jt, ht, mt = [], [], []
    jr, hr, mr = [], [], []
    for i in range(sample_count):
        tx, ty, tz = test_samples[i]
        tr, tg, tb = xyz_to_rgb(tx, ty, tz)
        trc, tgc, tbc = von_kries_adapt(tr, tg, tb, *test_rgb)
        trp, tgp, tbp = rgb_to_xyz(trc, tgc, tbc)
        trp, tgp, tbp = hunt_pointer_estevez(trp, tgp, tbp)
        tra, tga, tba = luminance_adaptation(trp, tgp, tbp)
        taw, tbw, ta = channel_ab(tra, tga, tba)
        
        rx, ry, rz = ref_samples[i]
        rr, rg, rb = xyz_to_rgb(rx, ry, rz)
        rrc, rgc, rbc = von_kries_adapt(rr, rg, rb, *ref_rgb)
        rrp, rgp, rbp = rgb_to_xyz(rrc, rgc, rbc)
        rrp, rgp, rbp = hunt_pointer_estevez(rrp, rgp, rbp)
        rra, rga, rba = luminance_adaptation(rrp, rgp, rbp)
        raw, rbw, ra = channel_ab(rra, rga, rba)
        
        at = channel_ab(test_ra, test_ga, test_ba)[2]
        ar = channel_ab(ref_ra, ref_ga, ref_ba)[2]
        jt_val = 100 * (ta / at)**(0.69 * 1.9272) if at else 0.0
        jr_val = 100 * (ra / ar)**(0.69 * 1.9272) if ar else 0.0
        
        h_angle_t = np.degrees(np.arctan2(tbw, taw)) if (taw or tbw) else 0.0
        h_angle_t = h_angle_t if h_angle_t >= 0 else h_angle_t + 360
        h_angle_r = np.degrees(np.arctan2(rbw, raw)) if (raw or rbw) else 0.0
        h_angle_r = h_angle_r if h_angle_r >= 0 else h_angle_r + 360
        
        et_t = 0.25 * (np.cos(np.radians(h_angle_t) + 2) + 3.8)
        et_r = 0.25 * (np.cos(np.radians(h_angle_r) + 2) + 3.8)
        
        den_t = tra + tga + 21/20 * tba
        t_t = (50000/13 * 1.0003 * et_t * np.hypot(taw, tbw)) / den_t if den_t else 0.0
        
        den_r = rra + rga + 21/20 * rba
        t_r = (50000/13 * 1.0003 * et_r * np.hypot(raw, rbw)) / den_r if den_r else 0.0
        
        c_t = t_t**0.9 * np.sqrt(jt_val / 100) * (1.64 - 0.29**0.2)** 0.73 if jt_val >= 0 else 0.0
        c_r = t_r**0.9 * np.sqrt(jr_val / 100) * (1.64 - 0.29**0.2)** 0.73 if jr_val >= 0 else 0.0
        
        mt_val = c_t * 0.7937**0.25
        mr_val = c_r * 0.7937**0.25
        
        jt.append(jt_val)
        ht.append(h_angle_t)
        mt.append(mt_val)
        jr.append(jr_val)
        hr.append(h_angle_r)
        mr.append(mr_val)
    
    jpt_uc = [(1 + 100*0.007)*j/(1 + 0.007*j) if j else 0 for j in jt]
    jpr_uc = [(1 + 100*0.007)*j/(1 + 0.007*j) if j else 0 for j in jr]
    mpt_uc = [np.log(1 + 0.0228*m)/0.0228 if m >=0 else 0 for m in mt]
    mpr_uc = [np.log(1 + 0.0228*m)/0.0228 if m >=0 else 0 for m in mr]
    
    apt_uc = [mpt_uc[i] * np.cos(np.radians(ht[i])) for i in range(sample_count)]
    apr_uc = [mpr_uc[i] * np.cos(np.radians(hr[i])) for i in range(sample_count)]
    bpt_uc = [mpt_uc[i] * np.sin(np.radians(ht[i])) for i in range(sample_count)]
    bpr_uc = [mpr_uc[i] * np.sin(np.radians(hr[i])) for i in range(sample_count)]
    
    de = [round(np.sqrt((jpt_uc[i]-jpr_uc[i])**2 + (apt_uc[i]-apr_uc[i])** 2 + (bpt_uc[i]-bpr_uc[i])**2), 2) 
          for i in range(sample_count)]
    de_ave = sum(de)/sample_count if sample_count else 0.0
    
    rfi = [round(10 * np.log(np.exp((100 - 6.73*d)/10) + 1)) for d in de]
    rf = 10 * np.log(np.exp((100 - 6.73*de_ave)/10) + 1) if de_ave else 0.0
    
    bin_number = count_hue_angle(hr)
    jptj = np.zeros(16)
    aptj = np.zeros(16)
    bptj = np.zeros(16)
    jprj = np.zeros(16)
    aprj = np.zeros(16)
    bprj = np.zeros(16)
    per_bin_count = np.zeros(16, dtype=int)
    
    for i in range(sample_count):
        bin_idx = bin_number[i] - 1
        if 0 <= bin_idx < 16:
            per_bin_count[bin_idx] += 1
            jptj[bin_idx] += jpt_uc[i]
            aptj[bin_idx] += apt_uc[i]
            bptj[bin_idx] += bpt_uc[i]
            jprj[bin_idx] += jpr_uc[i]
            aprj[bin_idx] += apr_uc[i]
            bprj[bin_idx] += bpr_uc[i]
    
    for i in range(16):
        if per_bin_count[i] > 0:
            jptj[i] /= per_bin_count[i]
            aptj[i] /= per_bin_count[i]
            bptj[i] /= per_bin_count[i]
            jprj[i] /= per_bin_count[i]
            aprj[i] /= per_bin_count[i]
            bprj[i] /= per_bin_count[i]
    
    valid_bins = [i for i in range(16) if per_bin_count[i] > 0]
    area_test = polygon_area([aptj[i] for i in valid_bins], [bptj[i] for i in valid_bins])
    area_ref = polygon_area([aprj[i] for i in valid_bins], [bprj[i] for i in valid_bins])
    rg = 100 * area_test / area_ref if area_ref else 0.0
    
    return {
        'Rf': round(rf, 2),
        'Rg': round(rg, 2),
        'de_values': de,
        'test_a': apt_uc,
        'test_b': bpt_uc,
        'ref_a': apr_uc,
        'ref_b': bpr_uc,
        'hue_distribution': per_bin_count,
        'rfi_values': rfi,
        'cct_chebyshev': round(tc_chebyshev, 2),
        'cct_mccamy': round(tc_mccamy, 2)
    }

# 可视化函数
def plot_rf_rg(rf, rg):
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Rf (色彩保真度)', 'Rg (色域面积)']
    values = [rf, rg]
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3'], width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    ax.set_title('光源色品质评估指标', fontsize=16, pad=20)
    ax.set_ylim(0, 110)
    ax.set_ylabel('数值', fontsize=12)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.3)
    ax.text(0.5, 102, '参考基准(100)', ha='center', va='bottom', color='r', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_de_distribution(de_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(de_values, kde=True, ax=ax1, color='#FF9800', bins=15)
    ax1.set_title('色貌差(DE)分布', fontsize=14)
    ax1.set_xlabel('色貌差(DE)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.axvline(np.mean(de_values), color='r', linestyle='--', label=f'均值: {np.mean(de_values):.2f}')
    ax1.legend()
    
    sns.boxplot(y=de_values, ax=ax2, color='#FF9800')
    ax2.set_title('色貌差(DE)箱线图', fontsize=14)
    ax2.set_ylabel('色貌差(DE)', fontsize=12)
    ax2.text(0.1, np.mean(de_values), f'均值: {np.mean(de_values):.2f}', 
             ha='left', va='center', color='r')
    plt.tight_layout()
    return fig

def plot_hue_comparison(test_a, test_b, ref_a, ref_b, hue_distribution):
    fig, ax = plt.subplots(figsize=(10, 10))
    test_points = np.column_stack((test_a, test_b))
    ref_points = np.column_stack((ref_a, ref_b))
    
    test_angles = np.arctan2(test_points[:,1], test_points[:,0])
    test_sorted = test_points[np.argsort(test_angles)]
    test_poly = Polygon(test_sorted, fill=True, alpha=0.4, color='#2196F3', label='待测光源')
    
    ref_angles = np.arctan2(ref_points[:,1], ref_points[:,0])
    ref_sorted = ref_points[np.argsort(ref_angles)]
    ref_poly = Polygon(ref_sorted, fill=True, alpha=0.4, color='#4CAF50', label='参考光源')
    
    ax.add_patch(test_poly)
    ax.add_patch(ref_poly)
    ax.scatter(test_points[:,0], test_points[:,1], color='#2196F3', s=30, alpha=0.7)
    ax.scatter(ref_points[:,0], ref_points[:,1], color='#4CAF50', s=30, alpha=0.7)
    
    ax.set_title('光源色调范围比较', fontsize=16, pad=20)
    ax.set_xlabel('a* (红绿色轴)', fontsize=12)
    ax.set_ylabel('b* (黄蓝色轴)', fontsize=12)
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_rf_vs_de(de_values, rfi_values):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(de_values, rfi_values, c=de_values, cmap='viridis', 
                         alpha=0.7, edgecolors='w', s=50)
    cbar = plt.colorbar(scatter)
    cbar.set_label('色貌差(DE)', fontsize=12)
    
    z = np.polyfit(de_values, rfi_values, 3)
    p = np.poly1d(z)
    sorted_de = np.sort(de_values)
    ax.plot(sorted_de, p(sorted_de), "r--")
    
    ax.set_title('色貌差(DE)与Rf贡献值的关系', fontsize=16, pad=20)
    ax.set_xlabel('色貌差(DE)', fontsize=12)
    ax.set_ylabel('Rf贡献值', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

# 主函数
def main():
    result = calculate_rf_rg_with_visual_data()
    print(f"计算结果: Rf={result['Rf']}, Rg={result['Rg']}")
    
    # 生成并保存图表
    fig1 = plot_rf_rg(result['Rf'], result['Rg'])
    fig2 = plot_de_distribution(result['de_values'])
    fig3 = plot_hue_comparison(
        result['test_a'], result['test_b'],
        result['ref_a'], result['ref_b'],
        result['hue_distribution']
    )
    fig4 = plot_rf_vs_de(result['de_values'], result['rfi_values'])
    
    fig1.savefig('rf_rg_metrics.png', dpi=300, bbox_inches='tight')
    fig2.savefig('de_distribution.png', dpi=300, bbox_inches='tight')
    fig3.savefig('hue_comparison.png', dpi=300, bbox_inches='tight')
    fig4.savefig('rf_vs_de.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()