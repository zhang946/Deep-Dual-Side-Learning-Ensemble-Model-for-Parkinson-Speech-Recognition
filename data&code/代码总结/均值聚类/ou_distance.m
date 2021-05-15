function output = ou_distance(data, center)
% 变量data_num中存放的是样本数量
data_num = size(data, 1);
% 变量center_num中存放的是类别数
center_num = size(center, 1);
% 初始化输出变量
output = zeros(data_num, center_num);
% 求输出变量
for i = 1: center_num
    % 求样本集与第i个聚类中心的差值
    % 函数repmat的功能是复制矩阵，若B = repmat(A, m, n)，则认为B是由m×n块矩阵A平铺而成
    difference = data - repmat(center(i,:), data_num, 1);
    % 计算欧氏距离的平方
    output(:,i) = sum(difference .* difference, 2);
end