clc;
clear all;
% feature =xlsread('l1正则_lsvt.xlsx','B2:OU127');
feature =xlsread('l1正则_t2.xlsx','B2:DW1041');
% 对样本特征进行0-1归一化操作
flattened_feature = feature(:)';
mapped_flattened_feature = mapminmax(flattened_feature, 0, 1);
feature = reshape(mapped_flattened_feature, size(feature));
% 变量K中存放的是类别数，本例中K=2
K = 2;
% 从变量feature中随机挑选K个样本作为初始簇中心
% 变量data_num中存放的是样本数量
% 变量temp中存放的是随机产生的K个序号
% 变量center中存放的是挑选出的K个簇中心
data_num = size(feature, 1);
temp = randperm(data_num, K)';
center = feature(temp, :);
% 变量iteration中存放的是迭代次数
iteration = 0;
% 开始迭代
while 1
    % 变量distance中存放的是样本特征集与所有簇中心的欧氏距离的平方
    % 它是一个M×K的矩阵，M是样本量，K是类别数
    distance = ou_distance(feature, center);
    % 对变量distance的每一行从小到大排序，变量index中存放的是排序后的序号
    [~, index] = sort(distance, 2, 'ascend');
    % 计算新的簇中心
    center_new = zeros(K, size(feature, 2));
    for i = 1:K
        class_i_feature = feature(index(:, 1) == i, :);
        center_new(i, :) = mean(class_i_feature, 1);
    end
    % 更新迭代次数
    iteration = iteration + 1;
    % 输出当前迭代次数
    fprintf('当前迭代次数为：%d\n', iteration);
    % 如果聚类中心与上一次迭代相同，则停止迭代，跳出循环
    if center_new == center
        break;
    end
    % 否则用新的簇中心来取代旧的
    center = center_new;
end
% 变量result中存放的是最终的聚类结果
result = index(:, 1);
for i=1:1040
    if(result(i)==2)
        result(i)=1;
    else
        result(i)=0;
    end
end

xlswrite('label.xlsx',result);
