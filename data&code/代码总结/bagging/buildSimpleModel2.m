%*************生成一次弱分类器********%
function[correctReat,predit_label,label_test]=buildSimpleModel2(data,label)
%*******初始化所有样本属性集、标签集**********%
% data_attribute=data;
% data_label=label;
%% *******初始化训练样本属性集、标签集**********%

% data_u=[data,label];
% [row,col]=size(data_u);
% train_num=floor(0.8*length(data_u));
% train_data=data_u(randi(row,train_num,1),:);
% train_data_attribute=train_data(:,1:col-1);
% train_data_label=train_data(:,col);

%t2
% j=randi([1,1040],1);
% data_test=data(j,:);
% label_test=label(j,:);
% 
% data_u=data;
% label_u=label;
% 
% data_u(j,:)=[];
% label_u(j,:)=[];
% 
% data_train=data_u;
% label_train=label_u;

%lsvt
j=randi([1,126],1);
data_test=data(j,:);
label_test=label(j,:);

data_u=data;
label_u=label;

data_u(j,:)=[];
label_u(j,:)=[];

data_train=data_u;
label_train=label_u;
%*******获得模型**********%
model=libsvmtrain(label_train,data_train,'-c 12 -g 0.8 -t 2');
%*******获得预测分类标签**********%
predit_label=libsvmpredict(label_test,data_test, model);
%*******获得分类正确率**********%
correctReat=sum(predit_label==label_test)/size(data_test,1);


