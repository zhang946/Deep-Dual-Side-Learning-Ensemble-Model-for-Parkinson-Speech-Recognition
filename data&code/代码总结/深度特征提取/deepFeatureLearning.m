%%%%深度特征提取%%%%%%%%%%%
%参数说明：trainX训练样本，trainY训练标签；testX测试样本，testY测试标签
clear ; close all; clc
%load ('LSVT_voice_rehabilitation.mat');%.mat文件是数据集，存储trainX，trainY, testX,testY
% trainX=xlsread('LSVT_voice_rehabilitation.xlsx','B2:KY51');
% trainY=xlsread('LSVT_voice_rehabilitation.xlsx','A2:A51');
% testX=xlsread('LSVT_voice_rehabilitation.xlsx','B2:KY127');
% testY=xlsread('LSVT_voice_rehabilitation.xlsx','A2:A127');

% data=xlsread('帕金森总数据集_total2.xlsx','C7:AB1046');
% label=xlsread('帕金森总数据集_total2.xlsx','B7:B1046');
data=xlsread('LSVT_voice_rehabilitation.xlsx','B2:KY127');
label=xlsread('LSVT_voice_rehabilitation.xlsx','A2:A127');
%----------------测试集和训练集的划分---------------------------------------
% 测试数据占全部数据的比例
testRatio = 0.3;

% 训练集索引
trainIndices = crossvalind('HoldOut', size(data, 1), testRatio);
% 测试集索引
testIndices = ~trainIndices;

% 训练集和训练标签
trainX = data(trainIndices, :);
trainY = label(trainIndices, :);

% 测试集和测试标签
testX = data(testIndices, :);
testY = label(testIndices, :);
%-----------------深度特征提取————————————————————————
trainX_map = mapminmax(trainX',0,1);  %标准化
trainX = trainX_map';
teatX_map = mapminmax(testX',0,1); 
testX = teatX_map';
T = constructT(trainY);  
record = [];
best_accuracy = 1;
[m,n]= size(trainX);
[m1,n1]= size(testX);
%训练编码器：层数，各层神经单元数目以及相关的参数可根据具体的数据集调整
%这里是三层编码器
for i = 500:100:800
    for j = 200:100:400
        for k = 40:20:100
                hiddenSize = i;
                autoenc1 = trainAutoencoder(trainX',hiddenSize,...%编码器输入数据列数是样本数，行数是特征维数
                    'MaxEpochs',1000,...                          %迭代次数
                    'L2WeightRegularization',0.001,...            %正则系数
                    'SparsityRegularization',4,...                %稀疏约束
                    'SparsityProportion',0.05,...                 %稀疏比例
                    'DecoderTransferFunction','purelin');         %转移函数
                %%%具体参数说明可以参看matlab文档
                %Extract the features in the hidden layer.
                features1 = encode(autoenc1,trainX');

                hiddenSize = j;
                autoenc2 = trainAutoencoder(features1,hiddenSize,...
                     'MaxEpochs',1000,...
                    'L2WeightRegularization',0.001,...
                    'SparsityRegularization',4,...
                    'SparsityProportion',0.05,...
                    'DecoderTransferFunction','purelin');
                %Extract the features in the hidden layer.

                features2 = encode(autoenc2,features1);

                hiddenSize = k;
                autoenc3 = trainAutoencoder(features2,hiddenSize,...
                     'MaxEpochs',1000,...
                    'L2WeightRegularization',0.001,...
                    'SparsityRegularization',4,...
                    'SparsityProportion',0.05,...
                    'DecoderTransferFunction','purelin');
                %Extract the features in the hidden layer.
                features3 = encode(autoenc3,features2);
       
                softnet = trainSoftmaxLayer(features3,T,'LossFunction','crossentropy');  %增加分类层
                deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);                     %堆叠
                deepnet = train(deepnet,trainX',T);                                      %微调整个网络
                
                
                train_deepFeature = featureExtract(deepnet,m, trainX);                   %用训练好的三层网络提取深度特征
                test_deepFeature = featureExtract(deepnet,m1, testX);   
                 
                model = svmtrain(trainY,train_deepFeature,'-s 0 -c 10^5 -t 0 -q'); %训练分类器
                svm_pred = svmpredict(testY,test_deepFeature,model); 
                accuracy = mean(double(svm_pred == testY)) * 100; 
                
                
%                 
%                 type = deepnet(testX');
%                 [max_value,predict] = max(type,[],1);
%                 accuracy = mean(double(predict' == testY)) * 100;  
                  %记录当前网络各层节点数目以及分类准确度，增加代码保存分类准确度高的特征作为提取结果
                 StructureAndAcc = [i,j,k,accuracy];                                
                 record  = [record;StructureAndAcc];                              
         end
     end
end
xlswrite('deep_future02.xlsx',test_deepFeature);
 