function [T] = constructT(Y)
% constructT��������ǩ����ת������Ӧ�ľ�����ʽ
%  Y�Ǳ�ǩ
type_num = size(unique(Y),1);%�����Ŀ
[m,n] = size(Y);
T = zeros(type_num,m);
for i = 1:type_num
   index = find(Y==i);
   c = size(index,1);
   for j = 1:c
       T(i,index(j)) = 1;
   end
end
 
end

