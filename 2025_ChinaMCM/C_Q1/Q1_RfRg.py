import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 计算代码
def load_data():
    # 读取待测光源SPD
    spd_data = pd.read_csv('data/Problem_1.csv')
    spd_wavelengths = spd_data['wavelength'].values
    spd_values = spd_data['SPD'].values
    
    # 读取10°视角CIE三刺激值
    cie_data = pd.read_csv('data/CIE_xyz_1964_10deg.csv')
    cie_wavelengths = cie_data['wavelength'].values
    cie_x = cie_data['x'].fillna(0).values
    cie_y = cie_data['y'].fillna(0).values
    cie_z = cie_data['z'].fillna(0).values
    
    # 读取99个颜色样品反射率
    color_samples = pd.read_csv('data/99_color_samples.csv')
    sample_wavelengths = color_samples['wavelength'].values
    samples = color_samples.filter(regex='sample').values.T  # 99x401矩阵
    
    return {
        'spd': (spd_wavelengths, spd_values),
        'cie': (cie_wavelengths, cie_x, cie_y, cie_z),
        'samples': (sample_wavelengths, samples)
    }

def numerical_sum(arr, fn):
    n = len(arr) - 1
    total = 0.0
    for i in range(len(arr)):
        if i == 0 or i == n:
            total += fn(arr[i], i)
        else:
            factor = 2 if i % 2 == 0 else 4
            total += factor * fn(arr[i], i)
    return total / 3

def generate_blackbody_spectrum(tc, wavelengths):
    h = 6.62607015e-34
    c = 299792458
    k = 1.380649e-23
    a = 2 * np.pi * h * c ** 2
    b = h * c / k
    
    p_lambda = []
    for wl in wavelengths:
        lambda_m = wl * 1e-9  # 转换为米
        exponent = b / (lambda_m * tc)
        if exponent > 700:  # 避免指数溢出
            p = 0.0
        else:
            p = a * (lambda_m ** -5) / (np.exp(exponent) - 1)
        p_lambda.append(p)
    
    # 归一化
    max_p = max(p_lambda) if p_lambda else 1.0
    return [p / max_p for p in p_lambda]

def light_source_stimulus(spectrum, cie_x, cie_y, cie_z):
    def integrand_x(i):
        return spectrum[i] * cie_x[i]
    
    def integrand_y(i):
        return spectrum[i] * cie_y[i]
    
    def integrand_z(i):
        return spectrum[i] * cie_z[i]
    
    indices = np.arange(len(spectrum))
    sum_y = numerical_sum(indices, lambda _, i: integrand_y(i))
    k = 100 / sum_y if sum_y != 0 else 1.0
    
    sum_x = numerical_sum(indices, lambda _, i: integrand_x(i))
    sum_z = numerical_sum(indices, lambda _, i: integrand_z(i))
    
    x_capital = k * sum_x
    y_capital = 100.0
    z_capital = k * sum_z
    
    return {
        'x_capital': x_capital,
        'y_capital': y_capital,
        'z_capital': z_capital,
        'k': k
    }

def get_chromaticity(x, y, z):
    sum_xyz = x + y + z
    if sum_xyz == 0:
        return 0.0, 0.0
    return x / sum_xyz, y / sum_xyz

def color_sample_stimulus(sample, spectrum, k, cie_x, cie_y, cie_z):
    def integrand_x(i):
        return sample[i] * spectrum[i] * cie_x[i]
    
    def integrand_y(i):
        return sample[i] * spectrum[i] * cie_y[i]
    
    def integrand_z(i):
        return sample[i] * spectrum[i] * cie_z[i]
    
    indices = np.arange(len(spectrum))
    x = k * numerical_sum(indices, lambda _, i: integrand_x(i))
    y = k * numerical_sum(indices, lambda _, i: integrand_y(i))
    z = k * numerical_sum(indices, lambda _, i: integrand_z(i))
    
    return x, y, z

def von_kries_adapt(r, g, b, rw, gw, bw):
    rc = 100 * r / rw if rw != 0 else 0
    gc = 100 * g / gw if gw != 0 else 0
    bc = 100 * b / bw if bw != 0 else 0
    return rc, gc, bc

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

def luminance_adaptation(rp, gp, bp):
    fl = 0.7937
    def adapt(v):
        val = (fl * v / 100) ** 0.42
        return 400 * val / (27.13 + val) + 0.1
    return adapt(rp), adapt(gp), adapt(bp)

def channel_ab(r, g, b):
    aw = r - 12/11 * g + 1/11 * b
    bw = 1/9 * (r + g - 2*b)
    aW = 1.0003 * (2*r + g + 1/20*b - 0.305)
    return aw, bw, aW

def count_hue_angle(hue_angles):
    return [int(angle / 22.5) + 1 for angle in hue_angles]

def polygon_area(a, b):
    if len(a) != len(b):
        raise ValueError("数组长度必须相同")
    n = len(a)
    a = a + [a[0]]
    b = b + [b[0]]
    area = 0.0
    for i in range(n):
        area += a[i] * b[i+1] - a[i+1] * b[i]
    return 0.5 * abs(area)

# 扩展计算函数，返回可视化所需的中间数据
def calculate_rf_rg_with_visual_data():
    data = load_data()
    spd_wavelengths, spd_values = data['spd']
    cie_wavelengths, cie_x, cie_y, cie_z = data['cie']
    sample_wavelengths, samples = data['samples']
    
    # 确保所有数据波长对齐
    if not np.array_equal(spd_wavelengths, cie_wavelengths) or not np.array_equal(spd_wavelengths, sample_wavelengths):
        raise ValueError("所有数据的波长必须一致")
    
    # 归一化待测光源SPD
    max_spd = max(spd_values) if spd_values.size > 0 else 1.0
    testing_spectrum = [spd / max_spd for spd in spd_values]
    
    # 生成参照光谱(3903K使用黑体光谱)
    reference_spectrum = generate_blackbody_spectrum(3903, spd_wavelengths)
    
    # 计算待测光源和参照光源的三刺激值
    test_stim = light_source_stimulus(testing_spectrum, cie_x, cie_y, cie_z)
    ref_stim = light_source_stimulus(reference_spectrum, cie_x, cie_y, cie_z)
    
    # 计算色品坐标
    test_x, test_y = get_chromaticity(test_stim['x_capital'], test_stim['y_capital'], test_stim['z_capital'])
    ref_x, ref_y = get_chromaticity(ref_stim['x_capital'], ref_stim['y_capital'], ref_stim['z_capital'])
    
    # 计算所有颜色样品的三刺激值
    sample_count = 99
    test_samples = []
    ref_samples = []
    
    for i in range(sample_count):
        sample = samples[i] if i < len(samples) else np.zeros_like(spd_values)
        tx, ty, tz = color_sample_stimulus(sample, testing_spectrum, test_stim['k'], cie_x, cie_y, cie_z)
        rx, ry, rz = color_sample_stimulus(sample, reference_spectrum, ref_stim['k'], cie_x, cie_y, cie_z)
        test_samples.append((tx, ty, tz))
        ref_samples.append((rx, ry, rz))
    
    # 转换为RGB并进行色适应
    test_rgb = xyz_to_rgb(test_stim['x_capital'], test_stim['y_capital'], test_stim['z_capital'])
    ref_rgb = xyz_to_rgb(ref_stim['x_capital'], ref_stim['y_capital'], ref_stim['z_capital'])
    
    test_rc, test_gc, test_bc = von_kries_adapt(*test_rgb, *test_rgb)
    ref_rc, ref_gc, ref_bc = von_kries_adapt(*ref_rgb, *ref_rgb)
    
    # 转换回XYZ
    test_xct, test_yct, test_zct = rgb_to_xyz(test_rc, test_gc, test_bc)
    ref_xcr, ref_ycr, ref_zcr = rgb_to_xyz(ref_rc, ref_gc, ref_bc)
    
    # Hunt-Pointer-Estevez变换
    test_rp, test_gp, test_bp = hunt_pointer_estevez(test_xct, test_yct, test_zct)
    ref_rp, ref_gp, ref_bp = hunt_pointer_estevez(ref_xcr, ref_ycr, ref_zcr)
    
    # 亮度适应
    test_ra, test_ga, test_ba = luminance_adaptation(test_rp, test_gp, test_bp)
    ref_ra, ref_ga, ref_ba = luminance_adaptation(ref_rp, ref_gp, ref_bp)
    
    # 处理每个样品
    jt, ht, mt = [], [], []
    jr, hr, mr = [], [], []
    
    for i in range(sample_count):
        # 待测光源下的样品
        tx, ty, tz = test_samples[i]
        tr, tg, tb = xyz_to_rgb(tx, ty, tz)
        trc, tgc, tbc = von_kries_adapt(tr, tg, tb, *test_rgb)
        trp, tgp, tbp = rgb_to_xyz(trc, tgc, tbc)
        trp, tgp, tbp = hunt_pointer_estevez(trp, tgp, tbp)
        tra, tga, tba = luminance_adaptation(trp, tgp, tbp)
        taw, tbw, ta = channel_ab(tra, tga, tba)
        
        # 参照光源下的样品
        rx, ry, rz = ref_samples[i]
        rr, rg, rb = xyz_to_rgb(rx, ry, rz)
        rrc, rgc, rbc = von_kries_adapt(rr, rg, rb, *ref_rgb)
        rrp, rgp, rbp = rgb_to_xyz(rrc, rgc, rbc)
        rrp, rgp, rbp = hunt_pointer_estevez(rrp, rgp, rbp)
        rra, rga, rba = luminance_adaptation(rrp, rgp, rbp)
        raw, rbw, ra = channel_ab(rra, rga, rba)
        
        # 计算CIECAM02参数
        at = channel_ab(test_ra, test_ga, test_ba)[2]
        ar = channel_ab(ref_ra, ref_ga, ref_ba)[2]
        
        jt_val = 100 * (ta / at) ** (0.69 * 1.9272) if at != 0 else 0.0
        jr_val = 100 * (ra / ar) ** (0.69 * 1.9272) if ar != 0 else 0.0
        
        h_angle_t = np.degrees(np.arctan2(tbw, taw)) if (taw != 0 or tbw != 0) else 0.0
        h_angle_t = h_angle_t if h_angle_t >= 0 else h_angle_t + 360
        h_angle_r = np.degrees(np.arctan2(rbw, raw)) if (raw != 0 or rbw != 0) else 0.0
        h_angle_r = h_angle_r if h_angle_r >= 0 else h_angle_r + 360
        
        et_t = 0.25 * (np.cos(np.radians(h_angle_t) + 2) + 3.8)
        et_r = 0.25 * (np.cos(np.radians(h_angle_r) + 2) + 3.8)
        
        denominator_t = tra + tga + 21/20 * tba
        t_t = (50000 / 13 * 1.0003 * et_t * np.sqrt(taw**2 + tbw**2)) / denominator_t if denominator_t != 0 else 0.0
        
        denominator_r = rra + rga + 21/20 * rba
        t_r = (50000 / 13 * 1.0003 * et_r * np.sqrt(raw**2 + rbw**2)) / denominator_r if denominator_r != 0 else 0.0
        
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
    
    # 转换到CAM02-UCS色坐标
    jpt_uc = [(1 + 100 * 0.007) * j / (1 + 0.007 * j) if j != 0 else 0.0 for j in jt]
    jpr_uc = [(1 + 100 * 0.007) * j / (1 + 0.007 * j) if j != 0 else 0.0 for j in jr]
    mpt_uc = [np.log(1 + 0.0228 * m) / 0.0228 if m >= 0 else 0.0 for m in mt]
    mpr_uc = [np.log(1 + 0.0228 * m) / 0.0228 if m >= 0 else 0.0 for m in mr]
    
    apt_uc = [mpt_uc[i] * np.cos(np.radians(ht[i])) for i in range(sample_count)]
    apr_uc = [mpr_uc[i] * np.cos(np.radians(hr[i])) for i in range(sample_count)]
    bpt_uc = [mpt_uc[i] * np.sin(np.radians(ht[i])) for i in range(sample_count)]
    bpr_uc = [mpr_uc[i] * np.sin(np.radians(hr[i])) for i in range(sample_count)]
    
    # 计算色貌差
    de = []
    for i in range(sample_count):
        delta = np.sqrt(
            (jpt_uc[i] - jpr_uc[i])**2 +
            (apt_uc[i] - apr_uc[i])** 2 +
            (bpt_uc[i] - bpr_uc[i])**2
        )
        de.append(round(delta, 2))
    
    de_ave = sum(de) / sample_count if sample_count > 0 else 0.0
    
    # 计算Rf
    rfi = [10 * np.log(np.exp((100 - 6.73 * d) / 10) + 1) for d in de]
    rfi = [round(r) for r in rfi]
    rf = 10 * np.log(np.exp((100 - 6.73 * de_ave) / 10) + 1) if de_ave != 0 else 0.0
    
    # 计算Rg
    bin_number = count_hue_angle(hr)
    jptj = [0.0] * 16
    aptj = [0.0] * 16
    bptj = [0.0] * 16
    jprj = [0.0] * 16
    aprj = [0.0] * 16
    bprj = [0.0] * 16
    per_bin_count = [0] * 16
    
    for i in range(sample_count):
        bin_idx = bin_number[i] - 1  # 转换为0索引
        if 0 <= bin_idx < 16:
            per_bin_count[bin_idx] += 1
            jptj[bin_idx] += jpt_uc[i]
            aptj[bin_idx] += apt_uc[i]
            bptj[bin_idx] += bpt_uc[i]
            jprj[bin_idx] += jpr_uc[i]
            aprj[bin_idx] += apr_uc[i]
            bprj[bin_idx] += bpr_uc[i]
    
    # 计算平均值
    for i in range(16):
        if per_bin_count[i] > 0:
            jptj[i] /= per_bin_count[i]
            aptj[i] /= per_bin_count[i]
            bptj[i] /= per_bin_count[i]
            jprj[i] /= per_bin_count[i]
            aprj[i] /= per_bin_count[i]
            bprj[i] /= per_bin_count[i]
    
    # 过滤空bin
    valid_bins = [i for i in range(16) if per_bin_count[i] > 0]
    aptj_valid = [aptj[i] for i in valid_bins]
    bptj_valid = [bptj[i] for i in valid_bins]
    aprj_valid = [aprj[i] for i in valid_bins]
    bprj_valid = [bprj[i] for i in valid_bins]
    
    area_test = polygon_area(aptj_valid, bptj_valid) if aptj_valid else 0.0
    area_ref = polygon_area(aprj_valid, bprj_valid) if aprj_valid else 0.0
    rg = 100 * area_test / area_ref if area_ref != 0 else 0.0
    
    # 返回所有可视化所需数据
    return {
        'Rf': round(rf, 2),
        'Rg': round(rg, 2),
        'de_values': de,
        'test_a': apt_uc,
        'test_b': bpt_uc,
        'ref_a': apr_uc,
        'ref_b': bpr_uc,
        'hue_distribution': per_bin_count,
        'rfi_values': rfi
    }

# ----------------------
# 可视化函数
# ----------------------
def plot_rf_rg(rf, rg):
    """绘制Rf和Rg指标柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Rf (色彩保真度)', 'Rg (色域面积)']
    values = [rf, rg]
    
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3'], width=0.6)
    
    # 添加数值标签
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
    """绘制色貌差(DE)分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 直方图
    sns.histplot(de_values, kde=True, ax=ax1, color='#FF9800', bins=15)
    ax1.set_title('色貌差(DE)分布', fontsize=14)
    ax1.set_xlabel('色貌差(DE)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.axvline(np.mean(de_values), color='r', linestyle='--', label=f'均值: {np.mean(de_values):.2f}')
    ax1.legend()
    
    # 箱线图
    sns.boxplot(y=de_values, ax=ax2, color='#FF9800')
    ax2.set_title('色貌差(DE)箱线图', fontsize=14)
    ax2.set_ylabel('色貌差(DE)', fontsize=12)
    ax2.text(0.1, np.mean(de_values), f'均值: {np.mean(de_values):.2f}', 
             ha='left', va='center', color='r')
    
    plt.tight_layout()
    return fig

def plot_hue_comparison(test_a, test_b, ref_a, ref_b, hue_distribution):
    """绘制待测光源与参考光源的色调范围比较"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制色域多边形
    test_points = np.column_stack((test_a, test_b))
    ref_points = np.column_stack((ref_a, ref_b))
    
    # 按角度排序点以确保多边形闭合正确
    test_angles = np.arctan2(test_points[:,1], test_points[:,0])
    test_sorted = test_points[np.argsort(test_angles)]
    test_poly = Polygon(test_sorted, fill=True, alpha=0.4, color='#2196F3', label='待测光源(3903K)')
    
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
    """绘制Rf贡献值与色貌差的关系"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(de_values, rfi_values, c=de_values, cmap='viridis', 
                         alpha=0.7, edgecolors='w', s=50)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('色貌差(DE)', fontsize=12)
    
    # 添加趋势线
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

# ----------------------
# 主函数：执行计算并生成可视化
# ----------------------
def main():
    # 执行计算并获取所有数据
    result = calculate_rf_rg_with_visual_data()
    
    # 生成图表
    fig1 = plot_rf_rg(result['Rf'], result['Rg'])
    fig2 = plot_de_distribution(result['de_values'])
    fig3 = plot_hue_comparison(
        result['test_a'], result['test_b'],
        result['ref_a'], result['ref_b'],
        result['hue_distribution']
    )
    fig4 = plot_rf_vs_de(result['de_values'], result['rfi_values'])
    
    # 保存图表
    fig1.savefig('rf_rg_metrics.png', dpi=300, bbox_inches='tight')
    fig2.savefig('de_distribution.png', dpi=300, bbox_inches='tight')
    fig3.savefig('hue_comparison.png', dpi=300, bbox_inches='tight')
    fig4.savefig('rf_vs_de.png', dpi=300, bbox_inches='tight')
    
    print(f"计算结果: Rf={result['Rf']}, Rg={result['Rg']}")
    print("图表已保存为PNG文件")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()