function deep_feature = featureExtract(deepnet,m, X)
%featureExtract 自动编码器的特征提取函数
%   deepnet 训练好的堆栈深度网络； m为需要编码的样本数目， X为编码数据（不含标签）         
%     b1 = repmat(deepnet.b{1,1}',m,1);
    b1 = repmat(deepnet.b{1}',m,1);
    encode1 = X*deepnet.IW{1,1}' + b1; %第一层输出
%     deep_feature = encode1;
% 
%     b2 = repmat(deepnet.b{2,1}',m,1);
    b2 = repmat(deepnet.b{2}',m,1);
    encode2 = encode1*deepnet.LW{2,1}' + b2; %第二层输出
%     deep_feature = encode2;

%     b3 = repmat(deepnet.b{3,1}',m,1);
    b3 = repmat(deepnet.b{3}',m,1);
    encode3 = encode2*deepnet.LW{3,2}' + b3;
    %第三层输出function [outputArg1,outputArg2] = untitled3(inputArg1,inputArg2)
       
%     b4 = repmat(deepnet.b{4,1}',m,1);
    b4 = repmat(deepnet.b{4}',m,1);
    encode4 = encode2*deepnet.LW{4,3}' + b4;
%     
        b5 = repmat(deepnet.b{5}',m,1);
    encode5 = encode2*deepnet.LW{5,4}' + b5;
    
     deep_feature = encode5;
end

