clc;
clear all;
close all;
%% 读取图像
class=10;
numberpclass=500; %每一类样本数 
ReadDir='./data/';%读取data文件夹的路径%视情况修改
for i=1:class
for j=1:numberpclass    
photo_name=[num2str(i-1,'%d'),'/',num2str(i-1,'%d'),num2str(j,'_%d'),'.bmp'];%图片名  
photo_index=[ReadDir,photo_name];%路径加图片名得到总的图片索引  
photo_matrix=((imread(photo_index)));%使用imread得到图像矩阵  
photo_matrix = imresize(photo_matrix, [5 7]);
photo_matrix =(dec2bin(photo_matrix(:))-'0')';
photo_feature(:,(i-1)*numberpclass+j)=photo_matrix(:); 
end
end
ann_data=photo_feature;              % 将特征矩阵传入到ann_data
ann_data(ann_data==0)=-1;%图像二值化
ann_label=zeros(1,numberpclass*class);      % 构造一个零矩阵，size是10*5000大小
for i=1:class
  for j=numberpclass*(i-1)+1:numberpclass*i     
     ann_label(i,j)=1;      
  end
end

acc=zeros(10,5);   
mini_batch1=[100,450,500,900,1500,4500];
max_epoch1=[500,1000,1500,2000,3000,4000,5000,8000,10000];
lr1=[0.012,0.0115,0.0125,0.013,0.0135,0.014,0.0145];
midnum1=[600,700,900];
decay_epoch1=[500 550];
f=['b' 'g' 'r' 'k'  'c' 'm' 'y'];%蓝色 绿色 红色 黑色 青色 紫色 黄色
sigma1=[0,0.0001,0.001,0.01,0.1];
for j=1
       mini_batch=mini_batch1(4);
       max_epoch=max_epoch1(1);
       lr=lr1(1);
       midnum=midnum1(3);
       decay_epoch=decay_epoch1(1);
       f1=f(1);
       sigma=sigma1(j);
  for times=1:5
%% 选定训练集和测试集
k=rand(1,numberpclass*class);  % 产生一行5000列的0-1之间的行矩阵
[m,n]=sort(k);  % 随机选择样本% m是排序后返回的序列值，n是该值在原矩阵中的索引号，也就是序列号
ntraindata=4500;    % 4500个训练集
ntestdata=500;      % 500个测试集
train_data=ann_data(:,n(1:ntraindata)); % 特征矩阵的前4500列传到train_data变量里面去
test_data=ann_data(:,n(ntraindata+1:numberpclass*class));   % 将4501到5000个特征矩阵传到test_data里面去
train_label=ann_label(:,n(1:ntraindata));       % 将前4500个标签传入
test_label=ann_label(:,n(ntraindata+1:numberpclass*class)); %将后500个标签传入
%% BP神经网络创建，训练和测试
%net=network_train(train_data,train_label);
    [w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk]=train_net1(train_data,train_label,mini_batch,f1,max_epoch,lr,midnum,decay_epoch,sigma);
%predict_label=network_test(test_data,net);
    predict_label=test_net1(test_data,w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk);
%% 测试样本正确率计算
[u,~]=find(test_label==1);
u=u';
for k=1:ntestdata
   if find(predict_label(:,k)>=0.9)
       c=find(predict_label(:,k)>=0.9);
       [~,d]=max(predict_label(:,k));
       x(:,k)=intersect(c,d);
   else 
       x(1,k)=100;
   end
end

error=abs(x-u);
accuracy=length(find(error==0))/length(error);
acc(j,times)=accuracy;
   
  end
 end

