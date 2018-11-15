%% MRS Test
%***/嘿嘿嘿这个是iris的数据集/***
%*****2018/7/4 Shijun Qiao*****
%    **********************
%       ┑(￣Д ￣)┍真难改
% 2018/7/23 Ar离子刻蚀15min 写19V 0.5s 读0.5V
%% 第一步:加载数据及函数目录
for time =1:10
for i=1:1801
addpath('G:\MRS iris');
 if i==1
%% 读取iris数据
iris = xlsread('iris.xlsx');
%提取数据,数据归一化,标签二值化
iris_data=iris(:,1:4);iris_label=iris(:,5:end);
iris_data_norm = bsxfun(@rdivide, bsxfun(@minus,iris_data,min(iris_data)), (max(iris_data) - min(iris_data)));
iris_data_norm=iris_data_norm';
iris_label=iris_label';
%分为训练集和测试集
ntrain=135;    % 训练集个数
ntest=15;      % 测试集个数
[Train, Test] = crossvalind('HoldOut', size(iris,1), 0.1);%选取10%数据作为test 
train_data = iris_data_norm(:,Train);          %训练集输入
train_label = iris_label(:,Train);        %训练集标记
test_data = iris_data_norm(:,Test);           %测试集输入
test_label = iris_label(:,Test);         %测试集标记
%数据二值化
for ii=1:150
    iris_data_norm_bin_mat=dec2bin(round(63*iris_data_norm(:,ii)),6)-'0';
    iris_data_norm_bin_mat(iris_data_norm_bin_mat==0)=-1;
    iris_data_norm_bin(:,ii)=iris_data_norm_bin_mat(iris_data_norm_bin_mat>=-1);
end
for ii=1:ntrain
    train_data_bin_mat=dec2bin(round(63*train_data(:,ii)),6)-'0';
    train_data_bin_mat(train_data_bin_mat==0)=-1;
    train_data_bin(:,ii)=train_data_bin_mat(train_data_bin_mat>=-1);
end
for ii=1:ntest
    test_data_bin_mat=dec2bin(round(63*test_data(:,ii)),6)-'0';
    test_data_bin_mat(test_data_bin_mat==0)=-1;
    test_data_bin(:,ii)=test_data_bin_mat(test_data_bin_mat>=-1);
end
%超参数备选
mini_batch1=[1,3,5,15,45,135];
max_epoch1=[50,100,200,500,800,1000];
lr1=[0.001,0.005,0.0075,0.01,0.05,0.1];
midnum1=[10,20,30,50,80,100];
decay_epoch1=[25];
f=['b' 'g' 'r' 'k'  'c' 'm' 'y'];%蓝色 绿色 红色 黑色 青色 紫色9 黄色
sigma1=[0,0.0001,0.001,0.01,0.1];
%网络参数
       mini_batch=mini_batch1(4);
       max_epoch=max_epoch1(3);
       lr=lr1(5);
       midnum=midnum1(3);
       decay_epoch=decay_epoch1(1);
       f1=f(1);
       sigma=sigma1(1);
       max_i=ntrain/mini_batch*max_epoch;
%忆阻器参数
       MRS_num=36;
%% BP神经网络创建，训练和测试
%net=network_train(train_data,train_label);
%% 三层神经网络的创建
innum = size(train_data_bin,1); %输入神经元个数(输入维数)
%midnum = 200; %隐层神经元个数
outnum = size(train_label,1); %输出神经元个数
% 权值初始化
w_ij0 = (rands(midnum, innum));
w_ki0 = (rands(outnum,midnum));
gammai=1.0000001;
gammak=1.0000001;
betai=0.0000001;
betak=0.0000001;
err_goal = 0.09; %误差目标最小值
lr0=lr;
[~,N] = size(train_data_bin); 
Oi = zeros(midnum,N);Ok = zeros(outnum,N); 
w_ij = binary(w_ij0);
w_ki = binary(w_ki0);
%记录权值初始化
    w_ki_cell=cell(max_i,1);
    ww=w_ki';
    w_ki_cell(i,1)={ww(:)};
E0=inf;
r=0.5;
copies=N/mini_batch;
record_muk=zeros(outnum,max_epoch*copies);
record_mui=zeros(midnum,max_epoch*copies);
m={0,0,0,0,0,0};v={0,0,0,0,0,0};delta=cell(1,6);
%初始化忆阻器阵列
loss=0;
for ii = 1:MRS_num
    Vpp(1,ii)={['Vpp' num2str(ii)]};
end
  w_MRS=w_ki';
for ii = 1:MRS_num
    Vpp_wave(1:500)=19*w_MRS(ii);
    Vpp_wave(501:1000)=0;
    Vpp_mat(1,ii)={Vpp_wave};
end
sample_rate=1000;
disp(['loss=[',mat2str(loss),'];']);
disp(['sample_rate=[',mat2str(sample_rate),'];']);
for ii = 1:MRS_num
    disp([Vpp{ii},'=',mat2str(Vpp_mat{1,ii}),';']);
end
%% 学习过程
else
        num=mod(i-1,copies)*mini_batch+1;
     %计算隐层神经元节点输出
         %w_ij = binary(w_ij0);%输入层到隐层的权值二值化  
         NETi(:,num:num+mini_batch-1) = w_ij*train_data_bin(:,num:num+mini_batch-1);
         [Oi(:,num:num+mini_batch-1),mui,xmui,xhati,vari,varti] = BN_train(NETi(:,num:num+mini_batch-1),gammai,betai);%输出标准化
         Oib(:,num:num+mini_batch-1)=binary(Oi(:,num:num+mini_batch-1));%输出二值化
    %计算输出层神经元节点输出
        %w_ki =binary(w_ki0);%隐层到输出层的权值二值化
        NETk(:,num:num+mini_batch-1) = w_ki*Oib(:,num:num+mini_batch-1);
        [Ok(:,num:num+mini_batch-1),muk,xmuk,xhatk,vark,vartk]=BN_train(NETk(:,num:num+mini_batch-1),gammak,betak);%输出标准化
        Ok(:,num:num+mini_batch-1)=1./(1+exp((-Ok(:,num:num+mini_batch-1))));  
    %调整输出层权重
    delta_k = (Ok(:,num:num+mini_batch-1)-train_label(:,num:num+mini_batch-1)).*Ok(:,num:num+mini_batch-1).*(1-Ok(:,num:num+mini_batch-1));
    [delta_NETk,dgammak,dbetak] = BBN(delta_k,xmuk,vark,gammak,xhatk);%标准化反向传播
    delta_i =  w_ki'*delta_NETk;
    delta_wk=delta_NETk*Oib(:,num:num+mini_batch-1)'/mini_batch;
    delta_i=delta_i.*(Oi(:,num:num+mini_batch-1)<=1 & Oi(:,num:num+mini_batch-1)>=-1);
    [delta_NETi,dgammai,dbetai] = BBN(delta_i,xmui,vari,gammai,xhati);
    delta_wi=delta_NETi*train_data_bin(:,num:num+mini_batch-1)'/mini_batch;
    %Adam
    grad={delta_wk,delta_wi,dgammak,dgammai,dbetak,dbetai};
    for j=1:6
    [delta{j},m{j},v{j}]=adam(grad{j},m{j},v{j},i);
    end
%    %更新存在误差
%    for j=1:6
%        [MM,NN]=size(delta{j});
%        delta{j}=delta{j}.*(1+(sigma/3)^2*randn(MM,NN));
%    end
    
    %更新参数
    w_ki0=clip(w_ki0-delta{1}*lr);
    w_ij0=clip(w_ij0-delta{2}*lr);
    
    w_ij = binary(w_ij0);
    w_ki1 = binary(w_ki0);
    %记算忆阻器权值应变量
    delta_w_ki=w_ki1-w_ki;
    w_ki=w_ki1;
    %记录权值变化过程与忆阻器阻值变化过程对照
    ww=w_ki';
    w_ki_cell(i,1)={ww(:)};
       
    gammak=gammak-delta{3}*lr;
    gammai=gammai-delta{4}*lr;
    
    betak=betak-delta{5}*lr;
    betai=betai-delta{6}*lr;
    
    %忆阻器更新阻值  输出任何波形由w变化决定
    %HRS—>LRS 正向电压   LRS->HRS 负向电压
    % + -19V 500ms 脉冲电压
    delta_w_MRS=delta_w_ki';
for ii = 1:MRS_num
    Vpp_wave(1:500)=10*delta_w_MRS(ii);
    Vpp_wave(501:1000)=0;
    Vpp_mat(1,ii)={Vpp_wave};
end
   
     %记录BN层均值和方差
     j=i-1;
     record_muk(:,j)=muk(:);
     record_mui(:,j)=mui(:);
     record_vartk(:,j)=vartk(:);
     record_varti(:,j)=varti(:);

      loss=0;
      sample_rate=1000;
      disp(['loss=[',mat2str(loss),'];']);
      disp(['sample_rate=[',mat2str(sample_rate),'];']);
      for ii = 1:MRS_num
          disp([Vpp{ii},'=',mat2str(Vpp_mat{1,ii}),';']);
      end
 end
 %% 每一批训练完计算误差函数，进行学习率衰减和学习数据重组
    if mod(i-1,copies)==0
        %学习数据重组
        k=rand(1,N);  
        [~,n]=sort(k);
        train_data_bin=train_data_bin(:,n(1:N));
        train_label=train_label(:,n(1:N));
        %误差函数计算
               
    end
    %学习率衰减    
    decay_epoch=25;
    if mod(i-1,decay_epoch*copies)==0
        lr=lr0*(0.7^(i-1/(decay_epoch*copies)));
    end
end
%test BN层用
mui=mean(record_mui,2);
muk=mean(record_muk,2);
varti=mean(record_varti,2);
vartk=mean(record_vartk,2);
%权值变化写入excel
 w_ki_xls = [w_ki_cell{:}];
  xlswrite('w_ki.xlsx',w_ki_xls);
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
accuracy(1,time)=length(find(error==0))/length(error);
end