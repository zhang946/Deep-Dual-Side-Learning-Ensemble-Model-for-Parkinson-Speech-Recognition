function [T] = constructT(Y)
% constructT（）将标签向量转换成相应的矩阵形式
%  Y是标签
type_num = size(unique(Y),1);%类别数目
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

