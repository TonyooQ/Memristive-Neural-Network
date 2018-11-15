clc;
clear all;
close all;
%% ��ȡͼ��
class=10;
numberpclass=500; %ÿһ�������� 
ReadDir='./data/';%��ȡdata�ļ��е�·��%������޸�
for i=1:class
for j=1:numberpclass    
photo_name=[num2str(i-1,'%d'),'/',num2str(i-1,'%d'),num2str(j,'_%d'),'.bmp'];%ͼƬ��  
photo_index=[ReadDir,photo_name];%·����ͼƬ���õ��ܵ�ͼƬ����  
photo_matrix=((imread(photo_index)));%ʹ��imread�õ�ͼ�����  
photo_matrix = imresize(photo_matrix, [5 7]);
photo_matrix =(dec2bin(photo_matrix(:))-'0')';
photo_feature(:,(i-1)*numberpclass+j)=photo_matrix(:); 
end
end
ann_data=photo_feature;              % �����������뵽ann_data
ann_data(ann_data==0)=-1;%ͼ���ֵ��
ann_label=zeros(1,numberpclass*class);      % ����һ�������size��10*5000��С
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
f=['b' 'g' 'r' 'k'  'c' 'm' 'y'];%��ɫ ��ɫ ��ɫ ��ɫ ��ɫ ��ɫ ��ɫ
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
%% ѡ��ѵ�����Ͳ��Լ�
k=rand(1,numberpclass*class);  % ����һ��5000�е�0-1֮����о���
[m,n]=sort(k);  % ���ѡ������% m������󷵻ص�����ֵ��n�Ǹ�ֵ��ԭ�����е������ţ�Ҳ�������к�
ntraindata=4500;    % 4500��ѵ����
ntestdata=500;      % 500�����Լ�
train_data=ann_data(:,n(1:ntraindata)); % ���������ǰ4500�д���train_data��������ȥ
test_data=ann_data(:,n(ntraindata+1:numberpclass*class));   % ��4501��5000���������󴫵�test_data����ȥ
train_label=ann_label(:,n(1:ntraindata));       % ��ǰ4500����ǩ����
test_label=ann_label(:,n(ntraindata+1:numberpclass*class)); %����500����ǩ����
%% BP�����紴����ѵ���Ͳ���
%net=network_train(train_data,train_label);
    [w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk]=train_net1(train_data,train_label,mini_batch,f1,max_epoch,lr,midnum,decay_epoch,sigma);
%predict_label=network_test(test_data,net);
    predict_label=test_net1(test_data,w_ij0,w_ki0,gammai,betai,mui,varti,gammak,betak,muk,vartk);
%% ����������ȷ�ʼ���
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

