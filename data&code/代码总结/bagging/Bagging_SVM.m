%****************数据预处理 *************%
clc;
clear;
%% lsvt l1正则

            %****L1正则化后的数据集*******%
data1=xlsread('l1正则_lsvt.xlsx','A2:CV127');
label1=xlsread('l1正则_lsvt.xlsx','OU2:OU127');
            %****第一次均值聚类后的数据集*******%
data2=xlsread('l1正则_lsvt_均值聚类1.xlsx','A2:CV127');
label2=xlsread('l1正则_lsvt_均值聚类1.xlsx','OU2:OU127');       
            %****第二次均值聚类后的数据集*******%
data3=xlsread('l1正则_lsvt_均值聚类2.xlsx','A2:CV127');
label3=xlsread('l1正则_lsvt_均值聚类2.xlsx','OU2:OU127');

%% t2 l1正则
% 
%             %****L1正则化后的数据集*******%
% data1=xlsread('l1正则_t2.xlsx','A2:CV1041');
% label1=xlsread('l1正则_t2.xlsx','DW2:DW1041');
%             %****第一次均值聚类后的数据集*******%
% data2=xlsread('l1正则_t2均_值聚类1.xlsx','A2:CV1041');
% label2=xlsread('l1正则_t2均_值聚类1.xlsx','DW2:DW1041');       
%             %****第二次均值聚类后的数据集*******%
% data3=xlsread('l1正则_t2均_值聚类2.xlsx','A2:CV1041');
% label3=xlsread('l1正则_t2均_值聚类2.xlsx','DW2:DW1041');

%% lsvt 普通

% data1=xlsread('l1正则_lsvt.xlsx','A2:CV127');
% label1=xlsread('l1正则_lsvt.xlsx','OU2:OU127');
% data2=data1;
% label2=label1;
% data3=data1;
% label3=label1;

%% t2普通

% data1=xlsread('l1正则_t2.xlsx','A2:CV1041');
% label1=xlsread('l1正则_t2.xlsx','DW2:DW1041');
% data2=data1;
% label2=label1;
% data3=data1;
% label3=label1;

iter=200;
correctArr=ones(iter,1);
correctM=[];
correctS=[];
lm=[];
ty=[];
num_p=0;
num_n=0;
num_p_t=0;
num_n_t=0;
p_l=0;
t_l=0;
for j=1:100
    data1_u(:,j)=data1(:,j);
    data2_u(:,j)=data2(:,j);
    data3_u(:,j)=data3(:,j);
    num_p=0;
    num_n=0;
    num_p_t=0;
    num_n_t=0;
    for i=1:iter
      [correctArr(i,1),p_l,t_l]=baggingTrainModel(data1_u,label1,data2_u,label2,data3_u,label3);
      if p_l==1
          num_n=num_n+1;
          if p_l==t_l
              num_n_t=num_n_t+1;
          end
      else
          num_p=num_p+1;
          if p_l==t_l
              num_p_t=num_p_t+1;
          end
      end
    end
    lm=[lm;(num_p_t/num_p)];
    ty=[ty;(num_n_t/num_n)];
    correctM=[correctM;mean(correctArr)];
    correctS=[correctS;std(correctArr)]
    correctArr=ones(iter,1);
end
disp(sum(correctM)/iter);
figure
plot(correctM,'r-*');
xlabel('迭代次数');
ylabel('准确率');
% xlswrite('200timesaccuracy.xlsx',correctM)
save('correctS_z_lsvt.mat','correctS');
save('correctM_z_lsvt_2.mat','correctM');