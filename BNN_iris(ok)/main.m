clc;
clear all;
close all;
dbstop if error
%% 读取iris数据
iris = xlsread('Iris.xlsx');
%提取数据,数据归一化,标签二值化
iris_data=iris(:,1:4);iris_label=iris(:,5:end);
iris_data_norm = bsxfun(@rdivide, bsxfun(@minus,iris_data,min(iris_data)), (max(iris_data) - min(iris_data)));
iris_data_norm=iris_data_norm';
iris_label=iris_label';
  for time=1:10
%分为训练集和测试集
ntrain=135;    % 训练集个数
ntest=15;      % 测试集个数
[Train, Test] = crossvalind('HoldOut', size(iris,1), 0.1);%选取10%数据作为test 
train_data = iris_data_norm(:,Train);          %训练集输入
train_label = iris_label(:,Train);        %训练集标记
test_data = iris_data_norm(:,Test);           %测试集输入
test_label = iris_label(:,Test);         %测试集标记
%数据二值化
for i=1:150
    iris_data_norm_bin_mat=dec2bin(round(63*iris_data_norm(:,i)),6)-'0';
    iris_data_norm_bin_mat(iris_data_norm_bin_mat==0)=-1;
    iris_data_norm_bin(:,i)=iris_data_norm_bin_mat(iris_data_norm_bin_mat>=-1);
end
for i=1:ntrain
    train_data_bin_mat=dec2bin(round(63*train_data(:,i)),6)-'0';
    train_data_bin_mat(train_data_bin_mat==0)=-1;
    train_data_bin(:,i)=train_data_bin_mat(train_data_bin_mat>=-1);
end
for i=1:ntest
    test_data_bin_mat=dec2bin(round(63*test_data(:,i)),6)-'0';
    test_data_bin_mat(test_data_bin_mat==0)=-1;
    test_data_bin(:,i)=test_data_bin_mat(test_data_bin_mat>=-1);
end

mini_batch1=[1,3,5,15,45,135];
max_epoch1=[50,100,200,300,500,800,1000];
lr1=[0.001,0.005,0.0075,0.01,0.05,0.1];
midnum1=[10,20,30,50,80,100];
decay_epoch1=[25];
f=['b' 'g' 'r' 'k'  'c' 'm' 'y'];%蓝色 绿色 红色 黑色 青色 紫色9 黄色
sigma1=[0,0.0001,0.001,0.01,0.1];
for i=1
       mini_batch=mini_batch1(4);
       max_epoch=max_epoch1(4);
       lr=lr1(5);
       midnum=midnum1(3);
       decay_epoch=decay_epoch1(1);
       f1=f(1);
       sigma=sigma1(1);
%% BP神经网络创建，训练和测试
%net=network_train(train_data,train_label);
    [w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk,err_train]=train_net1(train_data_bin,train_label,mini_batch,f1,max_epoch,lr,midnum,decay_epoch,sigma);
%predict_label=network_test(test_data,net);
    predict_label=test_net1(iris_data_norm_bin,w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk);
%% 测试样本正确率计算
[u,~]=find(iris_label==1);
u=u';
%for k=1:ntest
%   if find(predict_label(:,k)>=0.9)
%       c=find(predict_label(:,k)>=0.9);
%       [~,d]=max(predict_label(:,k));
%       x(:,k)=intersect(c,d);
%   else 
%       x(1,k)=100;
%   end
%end
%error=abs(x-u);
%accuracy=length(find(error==0))/length(error);
%acc(j,times)=accuracy;

[max_prob,label]=max(predict_label,[],1);
error=abs(label-u);
accuracy(i,time)=length(find(error==0))/length(error);
   
  end
 end

