%%%%���������ȡ%%%%%%%%%%%
%����˵����trainXѵ��������trainYѵ����ǩ��testX����������testY���Ա�ǩ
clear ; close all; clc
%load ('LSVT_voice_rehabilitation.mat');%.mat�ļ������ݼ����洢trainX��trainY, testX,testY
% trainX=xlsread('LSVT_voice_rehabilitation.xlsx','B2:KY51');
% trainY=xlsread('LSVT_voice_rehabilitation.xlsx','A2:A51');
% testX=xlsread('LSVT_voice_rehabilitation.xlsx','B2:KY127');
% testY=xlsread('LSVT_voice_rehabilitation.xlsx','A2:A127');

% data=xlsread('����ɭ�����ݼ�_total2.xlsx','C7:AB1046');
% label=xlsread('����ɭ�����ݼ�_total2.xlsx','B7:B1046');
data=xlsread('LSVT_voice_rehabilitation.xlsx','B2:KY127');
label=xlsread('LSVT_voice_rehabilitation.xlsx','A2:A127');
%----------------���Լ���ѵ�����Ļ���---------------------------------------
% ��������ռȫ�����ݵı���
testRatio = 0.3;

% ѵ��������
trainIndices = crossvalind('HoldOut', size(data, 1), testRatio);
% ���Լ�����
testIndices = ~trainIndices;

% ѵ������ѵ����ǩ
trainX = data(trainIndices, :);
trainY = label(trainIndices, :);

% ���Լ��Ͳ��Ա�ǩ
testX = data(testIndices, :);
testY = label(testIndices, :);
%-----------------���������ȡ������������������������������������������������
trainX_map = mapminmax(trainX',0,1);  %��׼��
trainX = trainX_map';
teatX_map = mapminmax(testX',0,1); 
testX = teatX_map';
T = constructT(trainY);  
record = [];
best_accuracy = 1;
[m,n]= size(trainX);
[m1,n1]= size(testX);
%ѵ���������������������񾭵�Ԫ��Ŀ�Լ���صĲ����ɸ��ݾ�������ݼ�����
%���������������
for i = 500:100:800
    for j = 200:100:400
        for k = 40:20:100
                hiddenSize = i;
                autoenc1 = trainAutoencoder(trainX',hiddenSize,...%��������������������������������������ά��
                    'MaxEpochs',1000,...                          %��������
                    'L2WeightRegularization',0.001,...            %����ϵ��
                    'SparsityRegularization',4,...                %ϡ��Լ��
                    'SparsityProportion',0.05,...                 %ϡ�����
                    'DecoderTransferFunction','purelin');         %ת�ƺ���
                %%%�������˵�����Բο�matlab�ĵ�
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
       
                softnet = trainSoftmaxLayer(features3,T,'LossFunction','crossentropy');  %���ӷ����
                deepnet = stack(autoenc1,autoenc2,autoenc3,softnet);                     %�ѵ�
                deepnet = train(deepnet,trainX',T);                                      %΢����������
                
                
                train_deepFeature = featureExtract(deepnet,m, trainX);                   %��ѵ���õ�����������ȡ�������
                test_deepFeature = featureExtract(deepnet,m1, testX);   
                 
                model = svmtrain(trainY,train_deepFeature,'-s 0 -c 10^5 -t 0 -q'); %ѵ��������
                svm_pred = svmpredict(testY,test_deepFeature,model); 
                accuracy = mean(double(svm_pred == testY)) * 100; 
                
                
%                 
%                 type = deepnet(testX');
%                 [max_value,predict] = max(type,[],1);
%                 accuracy = mean(double(predict' == testY)) * 100;  
                  %��¼��ǰ�������ڵ���Ŀ�Լ�����׼ȷ�ȣ����Ӵ��뱣�����׼ȷ�ȸߵ�������Ϊ��ȡ���
                 StructureAndAcc = [i,j,k,accuracy];                                
                 record  = [record;StructureAndAcc];                              
         end
     end
end
xlswrite('deep_future02.xlsx',test_deepFeature);
 