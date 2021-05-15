function output = ou_distance(data, center)
% ����data_num�д�ŵ�����������
data_num = size(data, 1);
% ����center_num�д�ŵ��������
center_num = size(center, 1);
% ��ʼ���������
output = zeros(data_num, center_num);
% ���������
for i = 1: center_num
    % �����������i���������ĵĲ�ֵ
    % ����repmat�Ĺ����Ǹ��ƾ�����B = repmat(A, m, n)������ΪB����m��n�����Aƽ�̶���
    difference = data - repmat(center(i,:), data_num, 1);
    % ����ŷ�Ͼ����ƽ��
    output(:,i) = sum(difference .* difference, 2);
end