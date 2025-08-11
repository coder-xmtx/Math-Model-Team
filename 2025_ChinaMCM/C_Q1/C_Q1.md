---
日期: 2025-08-07
tags:
  - 数学建模
---

> [!warning] 引言
> 如何在满足照明需求的同时，优化LED光源的光谱特性以实现有益的生理节律调节效应，成为一个亟待解决的重要问题

> [!important] 问题1数据
> 波长（$380$ ~ $780$）和对应SPD

----------------------------------------------------------------

# 问题1 基于给定的<ins>光源光谱功率分布</ins>(SPD)数据，建立计算模型，求解三类共五个核心参数

## 1.0 首先通过给定的SPD数据计算标准CIE XYZ
CIE XYZ的计算基于CIE标准观察者函数（CIE 1931 或者CIE 1964或者CIE 1976），这些函数描述了人眼对不同波长光的感知相应，计算公式如下：
$$\begin{cases}
	X=k\int_{\lambda}{I\left( \lambda \right) \cdot \bar{x}\left( \lambda \right) \cdot \mathrm{d}\lambda}\\
	Y=k\int_{\lambda}{I\left( \lambda \right) \cdot \bar{y}\left( \lambda \right) \cdot \mathrm{d}\lambda}\\
	Z=k\int_{\lambda}{I\left( \lambda \right) \cdot \bar{z}\left( \lambda \right) \cdot \mathrm{d}\lambda}\\
\end{cases}$$
其中，
$L\left(\lambda\right)$：光谱数据（波长$\lambda$对应的SPD）
$\bar{x}\left( \lambda \right) ,\bar{y}\left( \lambda \right) ,\bar{z}\left( \lambda \right)$：CIE标准观察者的三刺激值函数
$\mathrm{d}\lambda$：波长间隔（1nm）
$K$ 为归化系数：
$$k=\frac{100}{\sum_{\lambda}{I\left( \lambda \right) \bar{y}\left( \lambda \right) \Delta \lambda}}$$


进行计算：
由于我们只有间隔1nm的数据，所以对每个波长$\lambda$，将SPD数据$I\left(\lambda\right)$与对应的$\bar{x}\left( \lambda \right) ,\bar{y}\left( \lambda \right) ,\bar{z}\left( \lambda \right)$相乘，并乘以波长间隔$\Delta\lambda$(1nm)，然后对所有波长的结果求和，得到$X$，$Y$，$Z$
$$X=k\sum_{\lambda}{I\left( \lambda \right) \cdot \bar{x}\left( \lambda \right)}\cdot \Delta \lambda$$
$$Y=k\sum_{\lambda}{I\left( \lambda \right) \cdot \bar{y}\left( \lambda \right)}\cdot \Delta \lambda$$ 
$$Z=k\sum_{\lambda}{I\left( \lambda \right) \cdot \bar{z}\left( \lambda \right)}\cdot \Delta \lambda$$

算出来的是XYZ坐标，但是我们需要的是（x,y）色品坐标，所以要进行归一化：
$$x=\frac{X}{X+Y+Z}$$
$$y=\frac{Y}{X+Y+Z}$$

得到了色品坐标才可以进行下一步计算



### 所以，要从CIE官方网站获取三刺激值 [CIE](https://cie.co.at/)
其中，
CIE 1931标准（2°视场）仅覆盖人眼中央凹区域的锥体细胞，适合观察一小块物体时的颜色感知；CIE 1964 标准（10°视场）覆盖更更广的视场，适合手掌大小的物体时的颜色感知，**适合LED灯相关色温测量**。
但是，
相对来说，CIE 1931标准更经典一点，故我们采用CIE 1931的标准参照数据来进行计算，顺便也提供利用CIE 1964标准计算出的结果。

数据准备：`CIE_xyz_1964_10deg.csv`，`CIE_xyz_1931_2deg.csv`，`Problem_1.csv`


### 计算结果
CIE 1931标准：(0.3840445003795555,0.3767800565662821)
CIE 1964标准：(0.3875379505804716, 0.37337820391292564)


-----------------------------------------------------------------

## 1.1 颜色特性参数
### --- 相关色温(CCT) ---
> [!tip] 光源相关色温的概念
> 与光源色坐标最靠近的黑体色坐标点所对应的黑体温度

我采用W2和W4的方法，在张浩的论文中提到，这两种方法对于4000K左右的误差也不会太大，最后算出的结果是（采用CIE 1931标准）
W2：3903.6483 K
W4：3903.1779 K
两者方法得出的结果接近，故可认为真实值也在3903K左右浮动

#### - W1. 三角垂足插值法（原，张浩）
黑体的光谱功率分布可由普朗克公式获得：
$$S\left( \lambda \right) =\frac{c_1}{\lambda \left[ \exp \left( \frac{c_2}{\lambda T} \right) -1 \right]}$$
后面巴拉巴拉一大堆

#### - W2. 黑体轨迹的 Chebyshev 法（原，张浩）
CIE 1931的$\left(x,y\right)$坐标要转成CIE 1960 $\left(u,v\right)$坐标
$$u=\frac{4x}{-2x+12y+3}$$
$$v=\frac{6y}{-2x+12y+3}$$

设温度为 $T$ 的黑体的色品坐标为 $\left[ u\left( T \right) ,v\left( T \right) \right]$，满足
$$\min \left[ \underset{T_i<T<T_f}{\max}\left| c\left( T \right) -\frac{P\left( T \right)}{Q\left( T \right)} \right| \right] $$
式中 $c\left(T\right)$ 代表 $u\left(T\right)$ 或 $v\left(T\right)$
根据 Remes 的方法，当 $1000K\le T \le 15000K$时，
$$\bar{u}\left( T \right) =\frac{0.860117757+1.54118254 \times \,\,10^{-4}T+1.28641212\times 10^{-7}T^2}{1+8.42420235\times 10^{-4}T+7.08145163\times 10^{-7}T^2}$$
$$\bar{v}\left( T \right) =\frac{0.317398726+4.22806245 \times \,\,10^{-5}T+4.20481691\times 10^{-8}T^2}{1-2.89741816\times 10^{-5}T+1.61456053\times 10^{-7}T^2}$$
根据等温线垂直于黑体轨迹，得
$$\frac{u\left( T_c \right) -u_c}{v\left( T_c \right) -v_c}=-\frac{\mathrm{d}v\left( T_c \right)}{\mathrm{d}u\left( T_c \right)}$$
其中$\left(u_c,v_c\right)$是待求相关色温$T_c$的色品坐标，只需
$$\frac{\mathrm{d}u\left( T_c \right)}{\mathrm{d}T_c}\left| u\left( T_c \right) -u_c \right|+\frac{\mathrm{d}v\left( T_c \right)}{\mathrm{d}T_c}\left| v\left( T_c \right) -v_c \right|=0$$
可得相关色温$T_c$

#### - W3. 模拟黑体轨迹弧线法（原，张浩）
当色温为 $1667K$ ~ $25000K$ 时：
$$T=A+B\times d$$
该方法有个叫微倒角的东西，没听过好像也没数据，不采用

#### - W4. McCamy近似公式法（原，张浩）
由色品坐标$\left(x,y\right)$直接求相关色温的简便方法：
$$T=-437n^3+3601n^2-6861n+5514.31$$
其中
$$n=\frac{x-0.3320}{y-0.1858}$$

#### - W5. 牛顿迭代法（原，李月）
也是巴拉巴拉一大堆

### --- 距离普朗克轨迹的距离(色偏差，Duv) ---
> [!warning]
> |Duv|≤0.054 才有实际意义（原，李月）

$$Duv=\left[sign\left( v_s-v_t \right)\right] \sqrt{\left( u_t-u_s \right) ^2+\left( v_t-v_s \right) ^2}$$
其中，
sign为符号函数
$$sign=\begin{cases}
	1,\left( v_s-v_t \right) \ge 0\\
	-1,\left( v_s-v_t \right) <0\\
\end{cases}$$
待测光源的色品坐标 $\left(u_s,v_s\right)$
黑体轨迹上与待测光源点最近的点$\left(u_t,v_t\right)$

需要读取黑体曲线色品坐标数据，`black_body_locus.xls`，来源于书籍《LED器件选型与评价》康玉柱（著）

需要将CIE 1931色品坐标 $\left(x,y\right)$ 转换为 CIE 1960色品坐标 $\left(u,v\right)$
$$u=\frac{4x}{-2x+12y+3}$$
$$v=\frac{6y}{-2x+12y+3}$$

#### 结果
最后结果为-0.001024，满足|Duv|≤0.054，具有实际研究意义

----------------------------------------------------------

## 颜色还原参数
### --- 保真度指数(Rf) --- （新，TM-30-24）
参考文献（Royer）
使用99个标准颜色样本，覆盖现实世界中的常见颜色
对测试条件和参考条件下的所有 99 个 CIE 色彩坐标的差异幅度进行平均。将这个平均差异乘以一个缩放因子，然后从 100 中减去。应用对数转换，使数值不低于 0，从而将总范围限定在 0 到 100 之间。数值为 100 表示与参考条件完全匹配，数值越低，与参考条件的差异越大。

TM-30-24原文翻译
$R_{\mathrm{f}}^{\prime}$，该方法用于平均色彩保真度的测量，通过确定每个CES在测试光源和参考光源下的CAM02-UCS坐标之间的差异$(\Delta E_{Jab,i})$，然后计算这些色差的算术平均值。该平均值将乘以6.73的因子并从100中减去：
$$ \begin{array}{r}R_{\mathrm{f}}^{\prime} = 100 - 6.73\Big[\frac{1}{99}\sum\limits_{i = 1}^{99}\Big(\Delta E_{Jab,i}\Big)\Big] \end{array} $$
最后，比例应调整为最小 $R_{\mathrm{f}}$ 值为 $0$ ，以避免产生负数。最终 $R_{\mathrm{f}}$ 值的重新缩放应通过以下方式完成：
$$ R_{\mathrm{f}} = 10\ln \left[\exp \left(R_{\mathrm{f}}^{\prime} / 10\right) + 1\right] $$
如本文件所述，$R_{\mathrm{f}}$ 是平均色彩保真度的准确测量——测试光源和参考光源所呈现颜色的相似性。它解决了熟悉的CIE一般色彩渲染指数$R_{\mathrm{a}}$（CRI）许多已知的局限性。本文档中定义的$R_{\mathrm{f}}$与CIE 224:2017中记录的$R_{\mathrm{f}}$相同。$R_{\mathrm{f}}$的范围为0到100，数值越高表示与参考的相似性越强。它并不试图描述多色环境中平均感知色彩保真度，也不涉及与色彩记忆相关的其他效应。它也不是人类色彩偏好或自然感知的测量。因此，最大化$R_{\mathrm{f}}$并不一定对应于增加的可取性或实用性，或任何其他感知属性。两个具有相同$R_{\mathrm{f}}$值（除了100）的光源不一定会导致它们照亮的空间中物体的颜色外观相同，即使它们具有相同的色度。通过它本身，$R_{\mathrm{f}}$ 在值接近 100 时最具信息性，因为那时所有与参考光源的颜色偏移在定义上都是最小的。在较低的 $R_{\mathrm{f}}$ 值时，需要额外的措施来理解颜色是如何被偏移的。

缩放因子 $k$ 的确定是为了使得 187 种商业可用光源的平均 $R_{\mathrm{f}}$ 值（$R_{\mathrm{a}} \geq 60$）等于相同光源的平均 CIE $R_{\mathrm{a}}$ 值。然而，$R_{\mathrm{f}}$ 和 CIE $R_{\mathrm{a}}$ 是不同的，光源的 $R_{\mathrm{f}}$ 值可能高于或低于 CIE $R_{\mathrm{a}}$ 值。不同的光源即使 CIE $R_{\mathrm{a}}$ 值为 80，其 $R_{\mathrm{f}}$ 值也可能相差超过 30 分。特别是，增加红色色度的光源往往具有比 CIE $R_{\mathrm{a}}$ 值更高的 $R_{\mathrm{f}}$ 值。此外，之前为了最大化 CIE $R_{\mathrm{a}}$ 或达到某个阈值（如 80）而优化的光源，可能由于更广泛的颜色样本特征而具有较低的 $R_{\mathrm{f}}$ 值。可以以比 99 CES 更高的保真度呈现用于计算 CIE Ra 的八个颜色样本，但反之尚未得到证明。图 3-1 说明了在 CAM02-UCS 上分布的示例光源的颜色偏移。

#### 总体思路
--参数准备
待测光源的相关色温小于4000K，选择参照光源为同色温黑体光谱

--三刺激值计算
分别计算待测光源$\left(X_t,Y_t,Z_t\right)$和参照光源$\left(X_r,Y_r,Z_r\right)$的CIE 1964 10°视角三刺激值，以及99个标准色样再两种光源下的三刺激值$\left(X_{wt},Y_{wt},Z_{wt}\right)$和$\left(X_{wr},Y_{wr},Z_{wr}\right)$，计算公式在最上面。

--色空间转换流程
1. XYZ→CAT02 RGB：通过转换矩阵$M_{cat02}$实现：
$$\left( \begin{array}{c}
	R\\
	G\\
	B\\
\end{array} \right) =M_{cat02}\left( \begin{array}{c}
	X\\
	Y\\
	Z\\
\end{array} \right) $$

2. Von Kries色适应：将色样在待测/参照光源下的RGB值适应到光源白场：
$$R_c=100\cdot \frac{R}{R_w},G_c=100\cdot \frac{G}{G_w},B_c=100\cdot \frac{B}{B_w}$$
其中，$R_w$，$G_w$，$B_w$为光源白场的RGB值

3. 再通过逆矩阵 $M_{cat02}^{-1}$ 转回XYZ
4. Hunt-Pointer-Estevez变换：将XYZ转换为$R' G' B'$
$$\left( \begin{array}{c}
	R'\\
	G'\\
	B'\\
\end{array} \right) =M_{HPE}\left( \begin{array}{c}
	X\\
	Y\\
	Z\\
\end{array} \right) $$
5. 亮度适应，：对$R' G' B'$进行非线性变换
$$R_a=\frac{400\cdot \left( \frac{f_L\cdot R'}{100} \right) ^{0.42}}{27.13+\left( \frac{f_L\cdot R'}{100} \right) ^{0.42}}+0.1$$
$G'$、$B'$类似，$f_L=0.7937$

--色貌参数计算（CIECAM02模型）
- 无彩色相应$A$：$A=1.0003\cdot \left( 2R_a+G_a+0.05B_a-0.305 \right)$
- 明度$J$：$J=100\cdot \left( \frac{A}{A_w} \right) ^{0.69\times 1.9272}$（$A_w$为光源白场的$A$）
- 色调角$h$：$$h=\mathrm{arc}\tan 2\left( b,a \right) \times \frac{180}{\pi}$$
- 视彩度$M$：
	1. 色调系数$e$：$e = 0.25 \cdot (\cos(h_{\text{rad}} + 2) + 3.8)$（$h_{\text{rad}}$为弧度的色调角）
	2. 彩度因子$t$： $t = \frac{50000/13 \times 1.0003 \times e \times \sqrt{a_w^2 + b_w^2}}{R_a + G_a + (21/20)B_a}$
	3. 彩度$C$：$C = t^{0.9} \cdot \sqrt{J/100} \cdot (1.64 - 0.29^{0.2})^{0.73}$
	4. 视彩度$M$：$M = C \cdot 0.7937^{0.25}$

--CAM02-UCS色坐标转换
 将明度、视彩度、色调角转换为均匀色空间坐标$(J', a', b')$： 
 - 明度$J'$：$J' = \frac{(1 + 100 \times 0.007) \times J}{1 + 0.007 \times J}$
 - 视彩度$M'$：$M' = \frac{1}{0.0228} \cdot \ln(1 + 0.0228 \times M)$
 - 色品坐标：$a' = M' \cdot \cos(h_{\text{rad}})$，$b' = M' \cdot \sin(h_{\text{rad}})$

-- Rf（色彩逼真度指数）计算** 
- **色貌差$\Delta E$**：每个样品在待测与参照下的$(J', a', b')$欧氏距离： $\Delta E = \sqrt{(J'_{t} - J'_{r})^2 + (a'_{t} - a'_{r})^2 + (b'_{t} - b'_{r})^2}$
- **平均色貌差**：$\Delta E_{\text{ave}} = \frac{1}{99} \sum \Delta E$ 
- **Rf**： $Rf = 10 \cdot \ln\left( \exp\left( \frac{100 - 6.73 \times \Delta E_{\text{ave}}}{10} \right) + 1 \right)$

Rf的结果为91.79

### --- 色域指数(Rg) --- （新，TM-30-24）
参考文献（Royer）
色域指数是 TM-30 用于相对色域面积的度量，它近似表示所有色调的平均色饱和度变化。
计算Rg首先需要将 99 个 CIE 标准观察者分布到 16 个色调角箱中，以便计算稳定面积。将每个箱中所有样本的 (a', b') 坐标取平均值，形成测试条件和参考条件下各一个 16 边形的顶点。Rg是两个多边形面积之商（测试面积除以参考面积），再乘以 100。

TM-30-24原文翻译
$R_{\mathrm{g}}$ 是一个度量，表示每个色相角区间内 CES 的平均 $(a^{\prime},b^{\prime})$ 坐标所跨越的面积，$(a_{\mathrm{test},j}^{\prime},b_{\mathrm{test},j}^{\prime})$ 和 $(a_{\mathrm{ref},j}^{\prime},b_{\mathrm{ref},j}^{\prime})$。$J^{\prime}$ 坐标被丢弃，因此 $(a_{\mathrm{test},j}^{\prime},b_{\mathrm{test},j}^{\prime})$ 和 $(a_{\mathrm{ref},j}^{\prime},b_{\mathrm{ref},j}^{\prime})$ 坐标各自形成一个多边形。$R_{\mathrm{g}}$ 的计算方法是将两个多边形的面积 $(A_{t}$ 和 $A_{r}$ 分别) 的比率乘以 100：
$$ R_{\mathrm{g}} = 100\times \frac{A_{\mathrm{t}}}{A_{\mathrm{r}}} $$
$R_{\mathrm{g}}$ 计算的示意图见图 3-4。$R_{\mathrm{g}}$ 值为 100 表示，平均而言，测试源不会增加或减少色度与参考光源相比。 然而，它并不表示在测试光源和参考光源下所有颜色的饱和度都将相等。 $R_{\mathrm{g}}$ 值大于 100 表示与测试光源相比，整体平均饱和度增加，而 $R_{\mathrm{g}}$ 值小于 100 表示整体平均饱和度减少。 $R_{\mathrm{g}}$ 并不描述饱和度增加或减少的颜色；两个具有相同 $R_{\mathrm{g}}$ 值的光源可能会以不同的方式呈现颜色。

因为 $R_{\mathrm{g}}$ 使用了 CAM02-UCS，参考光源的区域在适用的色温范围内几乎是恒定的，如图 3-5 所示。随着参考光源随色温的变化，色域形状存在一些小差异。

$R_{\mathrm{f}}$ 和 $R_{\mathrm{g}}$ —本文所述的两个全球平均指标，分别捕捉颜色保真度和色域面积—量化了颜色再现的不同维度。$^{1}$ 色域面积的增加或减少必然需要降低颜色保真度；这两个值不能同时达到最大值。$R_{\mathrm{g}}$ 值没有整体最大值，但随着 $R_{\mathrm{f}}$ 的减少，可能的范围会增加，如图 3-6 所示。例如，如果希望保持 $R_{\mathrm{f}}$ 的值在 80 以上，则 $R_{\mathrm{g}}$ 的值大约限制在 80 到 120 的范围内。最大化（或最小化）$R_{\mathrm{g}}$ 不一定对应于更高的期望值。两个具有相同 $R_{\mathrm{f}}$ 和 $R_{\mathrm{g}}$ 值的光源——或任何类似的平均颜色保真度和色域面积的度量——不一定会在它们照亮的空间中导致相同的颜色外观，因为两者都无法区分色调。

图 3-6. $R_{\mathrm{g}}$ 与 $R_{\mathrm{f}}$ 的图。阴影区域表示对于名义上的白光源，组合值无法实现的估计区域。蓝色点表示 244 个真实的光谱功率分布 (SPD)，绿色点表示 157 个建模的光谱功率分布 (例如，CIE F 系列，CIE D 系列，实验计算的 LED SPD)，均如“光源颜色再现的测量回顾及用于表征颜色再现的双测量系统的考虑”中所述。

#### 具体思路
-- **1. 色调角Bin划分** 
将99个色样按参照光源下的色调角$h_r$ 分为16个Bin（每个Bin覆盖22.5°），即第 $j$ 个Bin对应色调角范围$(22.5(j-1), 22.5j]$ 
-- **2. 各Bin的平均坐标计算** 对每个Bin，计算待测光源与参照光源下色样的 $a', b'$ 平均值：
待测光源：$(\bar{a}'_{t,j}, \bar{b}'_{t,j}) = \frac{1}{N_j} \sum\limits_{i \in Bin_j} (a'_{t,i}, b'_{t,i})$-
参照光源：$(\bar{a}'_{r,j}, \bar{b}'_{r,j}) = \frac{1}{N_j} \sum\limits_{i \in Bin_j} (a'_{r,i}, b'_{r,i})$（$N_j$ 为第 $j$ 个Bin内的色样数量）
-- **3. 多边形面积计算** 分别计算16个平均坐标 $(\bar{a}', \bar{b}')$ 构成的多边形面积（采用鞋带公式）： 
参照光源面积 $S_r$：由 $(\bar{a}'_{r,1}, \bar{b}'_{r,1}), ..., (\bar{a}'_{r,16}, \bar{b}'_{r,16})$构成
待测光源面积 $S_t$：由 $(\bar{a}'_{t,1}, \bar{b}'_{t,1}), ..., (\bar{a}'_{t,16}, \bar{b}'_{t,16})$构成
鞋带公式：
对于顶点 $(a_1,b_1), ..., (a_n,b_n)$，面积 $S = 0.5 \cdot |\sum\limits_{i=1}^n (a_i b_{i+1} - a_{i+1} b_i)|$（$a_{n+1}=a_1, b_{n+1}=b_1$）
-- **4. Rg计算** $R_g = 100 \cdot \frac{S_t}{S_r}$

Rg的结果为106.08

------------------------------------------

## 生理节律效应参数
### --- 褪黑素日光照度比(mel-DER) ---
mel-DER描述光源对 ipRGC（视网膜光敏神经节细胞）的刺激强度，用于衡量其影响褪黑素的能力。是光源的黑视素通量（Melanopic Flux）与相同照度下标准日光（D65）的黑视素通量的比值。

❗此方法不一定正确
此方法有待考察，查阅的几篇文献都是用下面那个”更多提到的方法”
$$\text{mel-DER}=\frac{\Phi _{test}}{\Phi _{D65}}$$
其中，
$\Phi _{test}$是测试光源的黑视素通量（单位：W）
$\Phi_{D65}$是标准日光D65的黑视素通量（单位：W）

两条公式：
$$\Phi _{test}=\int_{380}^{780}{I\left( \lambda \right) \cdot S_{mel}\left( \lambda \right) \cdot \mathrm{d}\lambda}$$
其中，
$I\left(\lambda\right)$是测试光源的SPD，数据在`Problem_1.csv`
$S_{mel}\left( \lambda \right)$是黑视素灵敏度函数，数据在`S_mel.csv`

$$\Phi _{D65}=\int_{380}^{780}{I_{D65}\left( \lambda \right) \cdot S_{mel}\left( \lambda \right) \cdot \mathrm{d}\lambda}$$
其中，$I_{D65}\left( \lambda \right)$为D65日光的SPD,数据在`CIE_std_illum_D65.csv`

在这里只给了380nm到780nm每隔1nm的数据，所以计算方法改变：
$$\Phi _{test}=\sum_{\lambda}{I\left( \lambda \right) S_{mel}\left( \lambda \right)}$$
$$\Phi _{D65}=\sum_{\lambda}{I_{D65}\left( \lambda \right) S_{mel}\left( \lambda \right)}$$

#### 更多提到的方法（新，康玉柱）
首先，我们需要先计算M/P比率（黑视素与明视觉之间的比值），此指标用于评估光对人体生理节律和视觉功能的综合影响，这里简写为MPR
$$\mathrm{MPR}=\frac{832\int_{380}^{780}{\Phi _e\left( \lambda \right) \cdot S_{mel}\left( \lambda \right) \cdot \mathrm{d}\lambda}}{683\int_{380}^{780}{\Phi _e\left( \lambda \right) \cdot V\left( \lambda \right) \cdot \mathrm{d}\lambda}}$$
其中，
黑视素前置系数：832 lm/W
明视觉光谱效能最大值系数： 683 lm/W
$\Phi_e\left(\lambda\right)$：待测光源光谱曲线
$V\left( \lambda \right)$：明视觉光谱曲线
$S_{mel}\left( \lambda \right)$：黑视素敏感曲线（ipRGC相应曲线）

然后，mel-DER与MPR之间存在线性关系：
$$\text{mel-DER}=\frac{1000}{1.326\times 832}\mathrm{MPR}$$

最后得到的结果为 0.640847