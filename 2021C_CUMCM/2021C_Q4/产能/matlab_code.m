%1.qi,t＜1时最大的供货量
a=g-d;%a=供货量-订货量
mask=a<0;%把小于0元素转换为逻辑值1 (true)
g_mask=g;%复制g矩阵，防止被覆盖
g_mask(~mask)=NaN;%把g_mask非true的位置替换为NaN
row_max=nanmax(g_mask,[],2);%导出g_mask行向量最大值，忽略NaN值
%[row_ind,col_ind] = find(a<0);
%2.持续供货特征计算
log=d>0;
d_log=d;
d_log(~log)=NaN;
%result = (d_log(:, 1:end-1) + d_log(:, 2:end)) / 2;%输出相邻两个数的平均值
b=g./d;
% num= (b(:,1:end-1) >= 1) + (b(:,2:end) >= 1);%输出相邻两个数中大于等于1的个数，但含NaN
has_nan = isnan(b(:,1:end-1)) | isnan(b(:,2:end));%判断相邻元素对中是否存在NaN
valid_pairs_gt1 = (b(:,1:end-1) >= 1) + (b(:,2:end) >= 1);% 计算不含NaN的元素对中大于1的个数
result = valid_pairs_gt1;% 含NaN的位置输出NaN，否则输出计数结果
result(has_nan) = NaN;  % 用NaN标记含NaN的元素对