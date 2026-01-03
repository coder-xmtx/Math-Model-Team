import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import simpson
import copy
from matplotlib.patches import Polygon

# 设置中文字体和图像参数
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# --------------------------
# 1. 数据加载模块
# --------------------------
def load_data():
    """加载所有必要的数据文件"""
    # 加载5通道LED光谱数据
    spd_data = pd.read_csv('data/Problem_2.csv')
    wavelengths = spd_data['wavelength'].values
    channels = {
        'blue': spd_data['Blue'].values,
        'green': spd_data['Green'].values,
        'red': spd_data['Red'].values,
        'ww': spd_data['Warm White'].values,
        'cw': spd_data['Cold White'].values
    }
    
    # 加载CIE标准观察者函数(2°和10°)
    cie2 = pd.read_csv('data/CIE_xyz_1931_2deg.csv')
    cie10 = pd.read_csv('data/CIE_xyz_1964_10deg.csv')
    
    # 加载黑体轨迹数据(Duv计算用)
    import xlrd
    bb_data = xlrd.open_workbook('data/black_body_locus.xls')
    bb_table = bb_data.sheets()[0]
    black_locus = np.array([[bb_table.cell(i, j).value for j in range(3)] for i in range(bb_table.nrows)])
    
    # 加载melanopic数据
    mel_data = pd.read_csv('data/melanopicPhotopic.csv')
    
    # 加载99色样和日光数据(Rf/Rg计算用)
    color_samples = pd.read_csv('data/99_color_samples.csv')
    daylight_data = pd.read_csv('data/common_daylight_1nm.csv')
    
    return {
        'wavelengths': wavelengths,
        'channels': channels,
        'cie2': cie2,
        'cie10': cie10,
        'black_locus': black_locus,
        'mel_data': mel_data,
        'color_samples': color_samples,
        'daylight': daylight_data
    }

# --------------------------
# 2. 光谱合成模块
# --------------------------
def synthesize_spectrum(weights, channels, wavelengths):
    """根据权重合成总光谱"""
    # 权重归一化
    weights = np.array(weights)
    weights = weights / np.sum(weights) if np.sum(weights) != 0 else weights
    
    # 线性叠加各通道光谱
    spd_total = (weights[0] * channels['blue'] +
                 weights[1] * channels['green'] +
                 weights[2] * channels['red'] +
                 weights[3] * channels['ww'] +
                 weights[4] * channels['cw'])
    return wavelengths, spd_total

# --------------------------
# 3. 完整Rf/Rg计算模块（移植自Q1_RfRg.py）
# --------------------------
class RfRgCalculator:
    def __init__(self, data):
        self.data = data
        self.wls = data['wavelengths']
        # 加载10°和2°CIE数据
        self.cie10_wl = data['cie10']['wavelength'].values
        self.cie10_x = data['cie10']['x'].fillna(0).values
        self.cie10_y = data['cie10']['y'].fillna(0).values
        self.cie10_z = data['cie10']['z'].fillna(0).values
        self.cie2_wl = data['cie2']['wavelength'].values
        self.cie2_x = data['cie2']['x'].fillna(0).values
        self.cie2_y = data['cie2']['y'].fillna(0).values
        self.cie2_z = data['cie2']['z'].fillna(0).values
        # 加载99色样
        self.sample_wl = data['color_samples']['wavelength'].values
        self.samples = data['color_samples'].filter(regex='sample').values.T  # 99x波长数
        # 加载日光数据
        self.daylight_wl = data['daylight']['wavelength'].values
        self.daylight_v1 = data['daylight']['V1'].values
        self.daylight_v2 = data['daylight']['V2'].values
        self.daylight_v3 = data['daylight']['V3'].values

    def calculate_cct_chebyshev(self, x, y):
        """使用Chebyshev法计算相关色温"""
        denominator = -2 * x + 12 * y + 3
        u_c = 4 * x / denominator
        v_c = 6 * y / denominator

        def u_bar(T):
            numerator = 0.860117757 + 1.54118254e-4 * T + 1.28641212e-7 * T**2
            denominator = 1 + 8.42420235e-4 * T + 7.08145163e-7 * T**2
            return numerator / denominator

        def v_bar(T):
            numerator = 0.317398726 + 4.22806245e-5 * T + 4.20481691e-8 * T**2
            denominator = 1 - 2.89741816e-5 * T + 1.61456053e-7 * T**2
            return numerator / denominator

        def derivative(func, T, h=0.1):
            return (func(T + h) - func(T - h)) / (2 * h)

        def objective(T):
            u_t = u_bar(T)
            v_t = v_bar(T)
            du_dT = derivative(u_bar, T)
            dv_dT = derivative(v_bar, T)
            return du_dT * (u_t - u_c) + dv_dT * (v_t - v_c)

        try:
            return brentq(objective, 1000, 15000, xtol=1e-5)
        except ValueError:
            T_values = np.linspace(1000, 15000, 100)
            f_values = [objective(T) for T in T_values]
            for i in range(len(T_values)-1):
                if f_values[i] * f_values[i+1] <= 0:
                    return brentq(objective, T_values[i], T_values[i+1], xtol=1e-5)
            return 0

    def generate_blackbody_spectrum(self, tc):
        """生成黑体光谱"""
        h = 6.62607015e-34
        c = 299792458
        k = 1.380649e-23
        a = 2 * np.pi * h * c **2
        b = h * c / k
        
        p_lambda = []
        for wl in self.wls:
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

    def composed_daylight(self, tc):
        """生成重组日光光谱"""
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
        spectrum = self.daylight_v1 + m1 * self.daylight_v2 + m2 * self.daylight_v3
        
        # 确保光谱非负
        spectrum[spectrum < 0] = 0.0
        return spectrum

    def mix_blackbody_daylight(self, tc):
        """混合黑体和日光光谱"""
        srp = self.generate_blackbody_spectrum(tc)
        srd = self.composed_daylight(tc)
        
        # 按2°视角明度函数归一化
        sum_y_srp = simpson(srp * self.cie2_y, self.wls)
        srp = 100 * srp / sum_y_srp if sum_y_srp != 0 else srp
        sum_y_srd = simpson(srd * self.cie2_y, self.wls)
        srd = 100 * srd / sum_y_srd if sum_y_srd != 0 else srd
        
        # 线性混合
        tb, te = 4000, 5000
        return (tc - tb)/(te - tb) * srd + (te - tc)/(te - tb) * srp

    def light_source_stimulus(self, spectrum, cie_x, cie_y, cie_z):
        """计算光源三刺激值"""
        sum_y = simpson(spectrum * cie_y, self.wls)
        k = 100 / sum_y if sum_y != 0 else 1.0
        x = k * simpson(spectrum * cie_x, self.wls)
        z = k * simpson(spectrum * cie_z, self.wls)
        return {'x_capital': x, 'y_capital': 100.0, 'z_capital': z, 'k': k}

    def get_chromaticity(self, x, y, z):
        """计算色品坐标"""
        sum_xyz = x + y + z
        return (x/sum_xyz, y/sum_xyz) if sum_xyz != 0 else (0.0, 0.0)

    def color_sample_stimulus(self, sample, spectrum, k, cie_x, cie_y, cie_z):
        """计算样品三刺激值"""
        x = k * simpson(sample * spectrum * cie_x, self.wls)
        y = k * simpson(sample * spectrum * cie_y, self.wls)
        z = k * simpson(sample * spectrum * cie_z, self.wls)
        return x, y, z

    def von_kries_adapt(self, r, g, b, rw, gw, bw):
        """von Kries色适应转换"""
        rc = 100 * r / rw if rw != 0 else 0
        gc = 100 * g / gw if gw != 0 else 0
        bc = 100 * b / bw if bw != 0 else 0
        return rc, gc, bc

    def xyz_to_rgb(self, x, y, z):
        """XYZ转RGB (MCAT02矩阵)"""
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

    def rgb_to_xyz(self, r, g, b):
        """RGB转XYZ (逆MCAT02矩阵)"""
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

    def hunt_pointer_estevez(self, x, y, z):
        """Hunt-Pointer-Estevez变换"""
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

    def luminance_adaptation(self, rp, gp, bp):
        """亮度适应计算"""
        fl = 0.7937
        def adapt(v):
            val = (fl * v / 100) **0.42
            return 400 * val / (27.13 + val) + 0.1
        return adapt(rp), adapt(gp), adapt(bp)

    def channel_ab(self, r, g, b):
        """计算a*b*通道"""
        aw = r - 12/11 * g + 1/11 * b
        bw = 1/9 * (r + g - 2*b)
        aW = 1.0003 * (2*r + g + 1/20*b - 0.305)
        return aw, bw, aW

    def count_hue_angle(self, hue_angles):
        """色调角分箱"""
        return [int(angle / 22.5) + 1 for angle in hue_angles]

    def polygon_area(self, a, b):
        """计算多边形面积"""
        if len(a) != len(b):
            raise ValueError("数组长度必须相同")
        n = len(a)
        a = np.append(a, a[0])
        b = np.append(b, b[0])
        area = 0.0
        for i in range(n):
            area += a[i] * b[i+1] - a[i+1] * b[i]
        return 0.5 * abs(area)

    def calculate(self, test_spectrum):
        """完整计算Rf和Rg"""
        # 波长一致性检查
        if not (np.array_equal(self.wls, self.cie10_wl) and 
                np.array_equal(self.wls, self.cie2_wl) and 
                np.array_equal(self.wls, self.sample_wl) and
                np.array_equal(self.wls, self.daylight_wl)):
            raise ValueError("所有数据的波长必须一致")
        
        # 待测光谱归一化
        max_spd = max(test_spectrum) if test_spectrum.size else 1.0
        testing_spectrum = test_spectrum / max_spd
        
        # 计算相关色温（Chebyshev法）
        # 修正：从字典中提取三刺激值
        stimulus = self.light_source_stimulus(testing_spectrum, self.cie2_x, self.cie2_y, self.cie2_z)
        X = stimulus['x_capital']
        Y = stimulus['y_capital']
        Z = stimulus['z_capital']
        x, y = self.get_chromaticity(X, Y, Z)
        tc = self.calculate_cct_chebyshev(x, y)
        
        # 根据色温选择参照光谱
        if tc < 4000:
            reference_spectrum = self.generate_blackbody_spectrum(tc)
        elif tc > 5000:
            reference_spectrum = self.composed_daylight(tc)
        else:
            reference_spectrum = self.mix_blackbody_daylight(tc)
        
        # 计算光源三刺激值
        test_stim = self.light_source_stimulus(testing_spectrum, self.cie10_x, self.cie10_y, self.cie10_z)
        ref_stim = self.light_source_stimulus(reference_spectrum, self.cie10_x, self.cie10_y, self.cie10_z)
        
        # 计算色品坐标
        test_x, test_y = self.get_chromaticity(test_stim['x_capital'], test_stim['y_capital'], test_stim['z_capital'])
        ref_x, ref_y = self.get_chromaticity(ref_stim['x_capital'], ref_stim['y_capital'], ref_stim['z_capital'])
        
        # 计算99个色样的三刺激值
        sample_count = 99
        test_samples = []
        ref_samples = []
        for i in range(sample_count):
            sample = self.samples[i] if i < len(self.samples) else np.zeros_like(testing_spectrum)
            tx, ty, tz = self.color_sample_stimulus(sample, testing_spectrum, test_stim['k'], self.cie10_x, self.cie10_y, self.cie10_z)
            rx, ry, rz = self.color_sample_stimulus(sample, reference_spectrum, ref_stim['k'], self.cie10_x, self.cie10_y, self.cie10_z)
            test_samples.append((tx, ty, tz))
            ref_samples.append((rx, ry, rz))
        
        # 色适应转换与色貌模型计算
        test_rgb = self.xyz_to_rgb(test_stim['x_capital'], test_stim['y_capital'], test_stim['z_capital'])
        ref_rgb = self.xyz_to_rgb(ref_stim['x_capital'], ref_stim['y_capital'], ref_stim['z_capital'])
        
        test_rc, test_gc, test_bc = self.von_kries_adapt(*test_rgb, *test_rgb)
        ref_rc, ref_gc, ref_bc = self.von_kries_adapt(*ref_rgb, *ref_rgb)
        
        test_xct, test_yct, test_zct = self.rgb_to_xyz(test_rc, test_gc, test_bc)
        ref_xcr, ref_ycr, ref_zcr = self.rgb_to_xyz(ref_rc, ref_gc, ref_bc)
        
        test_rp, test_gp, test_bp = self.hunt_pointer_estevez(test_xct, test_yct, test_zct)
        ref_rp, ref_gp, ref_bp = self.hunt_pointer_estevez(ref_xcr, ref_ycr, ref_zcr)
        
        test_ra, test_ga, test_ba = self.luminance_adaptation(test_rp, test_gp, test_bp)
        ref_ra, ref_ga, ref_ba = self.luminance_adaptation(ref_rp, ref_gp, ref_bp)
        
        # 计算色貌参数（J, h, M）
        jt, ht, mt = [], [], []
        jr, hr, mr = [], [], []
        for i in range(sample_count):
            tx, ty, tz = test_samples[i]
            tr, tg, tb = self.xyz_to_rgb(tx, ty, tz)
            trc, tgc, tbc = self.von_kries_adapt(tr, tg, tb, *test_rgb)
            trp, tgp, tbp = self.rgb_to_xyz(trc, tgc, tbc)
            trp, tgp, tbp = self.hunt_pointer_estevez(trp, tgp, tbp)
            tra, tga, tba = self.luminance_adaptation(trp, tgp, tbp)
            taw, tbw, ta = self.channel_ab(tra, tga, tba)
            
            rx, ry, rz = ref_samples[i]
            rr, rg, rb = self.xyz_to_rgb(rx, ry, rz)
            rrc, rgc, rbc = self.von_kries_adapt(rr, rg, rb, *ref_rgb)
            rrp, rgp, rbp = self.rgb_to_xyz(rrc, rgc, rbc)
            rrp, rgp, rbp = self.hunt_pointer_estevez(rrp, rgp, rbp)
            rra, rga, rba = self.luminance_adaptation(rrp, rgp, rbp)
            raw, rbw, ra = self.channel_ab(rra, rga, rba)
            
            # 计算明度J
            at = self.channel_ab(test_ra, test_ga, test_ba)[2]
            ar = self.channel_ab(ref_ra, ref_ga, ref_ba)[2]
            jt_val = 100 * (ta / at)**(0.69 * 1.9272) if at else 0.0
            jr_val = 100 * (ra / ar)**(0.69 * 1.9272) if ar else 0.0
            
            # 计算色调角h
            h_angle_t = np.degrees(np.arctan2(tbw, taw)) if (taw or tbw) else 0.0
            h_angle_t = h_angle_t if h_angle_t >= 0 else h_angle_t + 360
            h_angle_r = np.degrees(np.arctan2(rbw, raw)) if (raw or rbw) else 0.0
            h_angle_r = h_angle_r if h_angle_r >= 0 else h_angle_r + 360
            
            # 计算彩度M
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
        
        # 计算色貌差DE
        jpt_uc = [(1 + 100*0.007)*j/(1 + 0.007*j) if j else 0 for j in jt]
        jpr_uc = [(1 + 100*0.007)*j/(1 + 0.007*j) if j else 0 for j in jr]
        mpt_uc = [np.log(1 + 0.0228*m)/0.0228 if m >=0 else 0 for m in mt]
        mpr_uc = [np.log(1 + 0.0228*m)/0.0228 if m >=0 else 0 for m in mr]
        
        apt_uc = [mpt_uc[i] * np.cos(np.radians(ht[i])) for i in range(sample_count)]
        apr_uc = [mpr_uc[i] * np.cos(np.radians(hr[i])) for i in range(sample_count)]
        bpt_uc = [mpt_uc[i] * np.sin(np.radians(ht[i])) for i in range(sample_count)]
        bpr_uc = [mpr_uc[i] * np.sin(np.radians(hr[i])) for i in range(sample_count)]
        
        de = [np.sqrt((jpt_uc[i]-jpr_uc[i])**2 + (apt_uc[i]-apr_uc[i])** 2 + (bpt_uc[i]-bpr_uc[i])**2) 
              for i in range(sample_count)]
        de_ave = sum(de)/sample_count if sample_count else 0.0
        
        # 计算Rf
        rfi = [10 * np.log(np.exp((100 - 6.73*d)/10) + 1) for d in de]
        rf = 10 * np.log(np.exp((100 - 6.73*de_ave)/10) + 1) if de_ave else 0.0
        
        # 计算Rg（色域面积比）
        bin_number = self.count_hue_angle(hr)
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
        area_test = self.polygon_area([aptj[i] for i in valid_bins], [bptj[i] for i in valid_bins])
        area_ref = self.polygon_area([aprj[i] for i in valid_bins], [bprj[i] for i in valid_bins])
        rg = 100 * area_test / area_ref if area_ref else 0.0
        
        return round(rf, 2), round(rg, 2)

# --------------------------
# 4. 光学参数计算模块
# --------------------------
class OpticCalculator:
    def __init__(self, data):
        self.data = data
        self.wls = data['wavelengths']
        self.cie2_x = data['cie2']['x'].values
        self.cie2_y = data['cie2']['y'].values
        self.cie2_z = data['cie2']['z'].values
        self.rf_rg_calculator = RfRgCalculator(data)  # 初始化完整Rf/Rg计算器

    def calculate_xyz(self, spd):
        """计算CIE XYZ三刺激值"""
        delta_lambda = 1
        sum_y = simpson(spd * self.cie2_y, self.wls)
        k = 100 / sum_y if sum_y != 0 else 1
        X = k * simpson(spd * self.cie2_x, self.wls)
        Y = k * simpson(spd * self.cie2_y, self.wls)
        Z = k * simpson(spd * self.cie2_z, self.wls)
        return X, Y, Z

    def calculate_xy(self, X, Y, Z):
        """计算色品坐标(x,y)"""
        sum_xyz = X + Y + Z
        x = X / sum_xyz if sum_xyz != 0 else 0
        y = Y / sum_xyz if sum_xyz != 0 else 0
        return x, y

    def calculate_cct_chebyshev(self, x, y):
        """使用Chebyshev法计算相关色温"""
        return self.rf_rg_calculator.calculate_cct_chebyshev(x, y)

    def calculate_duv(self, x, y):
        """计算Duv"""
        def xy_to_uv(x, y):
            denom = -2 * x + 12 * y + 3
            return 4*x/denom, 6*y/denom

        xp, yp = self.data['black_locus'][:, 1], self.data['black_locus'][:, 2]
        up, vp = xy_to_uv(xp, yp)
        ut, vt = xy_to_uv(x, y)
        
        delta_c = np.sqrt((ut - up)**2 + (vt - vp)** 2)
        n = np.argmin(delta_c)
        sign = 1 if vt - vp[n] >= 0 else -1
        return sign * delta_c[n]

    def calculate_melder(self, spd):
        """计算mel-DER"""
        mel = self.data['mel_data']
        melanopic_flux = 832 * simpson(spd * mel['ipRGC'].values, self.wls)
        photopic_flux = 683 * simpson(spd * mel['bright-vision-curve'].values, self.wls)
        mpr = melanopic_flux / photopic_flux if photopic_flux != 0 else 0
        return mpr / 832 * 1000 / 1.326

    def calculate_rf_rg(self, spd):
        """计算Rf和Rg（完整逻辑）"""
        return self.rf_rg_calculator.calculate(spd)

# --------------------------
# 5. 粒子群优化模块
# --------------------------
class PSOOptimizer:
    def __init__(self, data, scenario=1):
        self.data = data
        self.calculator = OpticCalculator(data)
        self.scenario = scenario  # 1:日间模式, 2:夜间模式
        
        # PSO参数
        self.pop_size = 40
        self.max_iter = 150
        self.w_ini, self.w_end = 0.9, 0.4
        self.c1, self.c2 = 2.5, 2.0
        self.v_max = 0.1
        
        # 初始化粒子群
        self.particles = self._init_particles()
        self.gbest = None
        self.gbest_fitness = float('inf')

    def _init_particles(self):
        """初始化粒子位置和速度"""
        particles = []
        for _ in range(self.pop_size):
            # 随机生成非负权重
            weights = np.random.rand(5)
            weights /= np.sum(weights)
            velocity = np.random.uniform(-self.v_max, self.v_max, 5)
            particles.append({
                'position': weights,
                'velocity': velocity,
                'pbest': weights.copy(),
                'pbest_fitness': float('inf')
            })
        return particles

    def _calculate_fitness(self, weights):
        """计算适应度(带惩罚项)"""
        # 合成光谱
        wls, spd = synthesize_spectrum(
            weights, self.data['channels'], self.data['wavelengths']
        )
        
        # 计算光学参数
        X, Y, Z = self.calculator.calculate_xyz(spd)
        x, y = self.calculator.calculate_xy(X, Y, Z)
        cct = self.calculator.calculate_cct_chebyshev(x, y)
        duv = self.calculator.calculate_duv(x, y)
        rf, rg = self.calculator.calculate_rf_rg(spd)
        melder = self.calculator.calculate_melder(spd)
        
        # 约束检查
        penalty = 0
        if self.scenario == 1:
            # 日间模式约束
            if not (5500 <= cct <= 6500):
                penalty += 1000
            if not (95 <= rg <= 105):
                penalty += 1000
            if rf <= 88:
                penalty += 1000
            # 目标函数: 最大化Rf → 最小化 -Rf
            fitness = -rf + penalty
        else:
            # 夜间模式约束
            if not (2500 <= cct <= 3500):
                penalty += 1000
            if rf < 80:
                penalty += 1000
            # 目标函数: 最小化mel-DER
            fitness = melder + penalty
            
        return {
            'fitness': fitness,
            'params': {
                'cct': cct,
                'duv': duv,
                'rf': rf,
                'rg': rg,
                'melder': melder,
                'weights': weights.copy()
            }
        }

    def optimize(self):
        """执行PSO优化"""
        fitness_history = []
        
        for iter in range(self.max_iter):
            # 计算惯性权重
            w = self.w_ini - (self.w_ini - self.w_end) * iter / self.max_iter
            
            for particle in self.particles:
                # 计算适应度
                result = self._calculate_fitness(particle['position'])
                current_fitness = result['fitness']
                current_params = result['params']
                
                # 更新个体最优
                if current_fitness < particle['pbest_fitness']:
                    particle['pbest'] = particle['position'].copy()
                    particle['pbest_fitness'] = current_fitness
                    particle['pbest_params'] = current_params
                
                # 更新全局最优
                if current_fitness < self.gbest_fitness:
                    self.gbest = particle['position'].copy()
                    self.gbest_fitness = current_fitness
                    self.gbest_params = current_params
            
            # 更新速度和位置
            for particle in self.particles:
                r1, r2 = np.random.rand(2)
                # 速度更新
                cognitive = self.c1 * r1 * (particle['pbest'] - particle['position'])
                social = self.c2 * r2 * (self.gbest - particle['position'])
                particle['velocity'] = w * particle['velocity'] + cognitive + social
                # 速度限制
                particle['velocity'] = np.clip(particle['velocity'], -self.v_max, self.v_max)
                # 位置更新
                particle['position'] += particle['velocity']
                # 确保非负
                particle['position'] = np.clip(particle['position'], 0, None)
                # 重新归一化
                particle['position'] /= np.sum(particle['position']) if np.sum(particle['position']) !=0 else 1
            
            # 记录历史
            fitness_history.append(self.gbest_fitness)
            if (iter + 1) % 10 == 0:
                print(f"迭代 {iter+1}/{self.max_iter}, 最优适应度: {self.gbest_fitness:.2f}")
        
        return self.gbest_params, fitness_history

# --------------------------
# 6. 结果可视化模块
# --------------------------
def visualize_results(params, data, scenario):
    """可视化优化结果"""
    # 1. 光谱图
    wls, spd = synthesize_spectrum(
        params['weights'], data['channels'], data['wavelengths']
    )
    plt.figure()
    plt.plot(wls, spd, label='合成光谱')
    plt.xlabel('波长 (nm)')
    plt.ylabel('相对光谱功率')
    plt.title('合成光谱分布' + ('(日间模式)' if scenario==1 else '(夜间模式)'))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f'spectrum_scenario_{scenario}.png')
    
    # 2. 参数表
    print("\n优化结果:")
    print(f"相关色温 (CCT): {params['cct']:.1f} K")
    print(f"Duv: {params['duv']:.6f}")
    print(f"保真度指数 (Rf): {params['rf']:.1f}")
    print(f"色域指数 (Rg): {params['rg']:.1f}")
    print(f"视黑素效率比 (mel-DER): {params['melder']:.3f}")
    print("通道权重:")
    channels = ['蓝光', '绿光', '红光', '暖白光', '冷白光']
    for i, (name, w) in enumerate(zip(channels, params['weights'])):
        print(f"  {name}: {w:.4f}")
    
    # 3. 权重饼图
    plt.figure()
    plt.pie(params['weights'], labels=channels, autopct='%1.1f%%')
    plt.title('通道权重分布' + ('(日间模式)' if scenario==1 else '(夜间模式)'))
    plt.savefig(f'weights_scenario_{scenario}.png')

# --------------------------
# 7. 主函数
# --------------------------
def main():
    # 加载数据
    data = load_data()
    
    # 场景1: 日间照明模式
    print("=== 开始日间照明模式优化 ===")
    pso_day = PSOOptimizer(data, scenario=1)
    day_params, day_history = pso_day.optimize()
    visualize_results(day_params, data, scenario=1)
    
    # 场景2: 夜间助眠模式
    print("\n=== 开始夜间助眠模式优化 ===")
    pso_night = PSOOptimizer(data, scenario=2)
    night_params, night_history = pso_night.optimize()
    visualize_results(night_params, data, scenario=2)
    
    # 绘制适应度曲线
    # 绘制适应度曲线
    plt.figure()
    plt.plot(day_history, label='日间模式')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.title('优化过程适应度变化-日间')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fitness_history-day.png')
    plt.show()

    plt.figure()
    plt.plot(night_history, label='夜间模式')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.title('优化过程适应度变化-夜间')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fitness_history-night.png')
    plt.show()

if __name__ == "__main__":
    main()