clc;
clear all;
% feature =xlsread('l1����_lsvt.xlsx','B2:OU127');
feature =xlsread('l1����_t2.xlsx','B2:DW1041');
% ��������������0-1��һ������
flattened_feature = feature(:)';
mapped_flattened_feature = mapminmax(flattened_feature, 0, 1);
feature = reshape(mapped_flattened_feature, size(feature));
% ����K�д�ŵ����������������K=2
K = 2;
% �ӱ���feature�������ѡK��������Ϊ��ʼ������
% ����data_num�д�ŵ�����������
% ����temp�д�ŵ������������K�����
% ����center�д�ŵ�����ѡ����K��������
data_num = size(feature, 1);
temp = randperm(data_num, K)';
center = feature(temp, :);
% ����iteration�д�ŵ��ǵ�������
iteration = 0;
% ��ʼ����
while 1
    % ����distance�д�ŵ������������������д����ĵ�ŷ�Ͼ����ƽ��
    % ����һ��M��K�ľ���M����������K�������
    distance = ou_distance(feature, center);
    % �Ա���distance��ÿһ�д�С�������򣬱���index�д�ŵ������������
    [~, index] = sort(distance, 2, 'ascend');
    % �����µĴ�����
    center_new = zeros(K, size(feature, 2));
    for i = 1:K
        class_i_feature = feature(index(:, 1) == i, :);
        center_new(i, :) = mean(class_i_feature, 1);
    end
    % ���µ�������
    iteration = iteration + 1;
    % �����ǰ��������
    fprintf('��ǰ��������Ϊ��%d\n', iteration);
    % ���������������һ�ε�����ͬ����ֹͣ����������ѭ��
    if center_new == center
        break;
    end
    % �������µĴ�������ȡ���ɵ�
    center = center_new;
end
% ����result�д�ŵ������յľ�����
result = index(:, 1);
for i=1:1040
    if(result(i)==2)
        result(i)=1;
    else
        result(i)=0;
    end
end

xlswrite('label.xlsx',result);
