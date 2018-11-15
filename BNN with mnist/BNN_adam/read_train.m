function imglist = read_train(root)
% ni为读取图片张数，n为文件夹数目
%========读取文件夹========%
out_Files = dir(root);%展开
tempind=0;              % 初始化一个temibd变量应该是用来计数之类的
imglist=cell(0);        %初始化一个空的cell矩阵
n=length(out_Files);    %应该是out_Files变量中元素的个数
%========读取文件========%
for i = 1:n;            % 从第一个元素开始一个个的访问每个元素（也就是结构体）
    if strcmp(out_Files(i).name,'.')|| strcmp(out_Files(i).name,'..')  % 结构体的名称为‘.’,或者‘..’则跳出for循环，否则继续for循环
    else
        rootpath=strcat(root,'/',out_Files(i).name);  %连接函数用来将字符串连接起来，在这里是得到需要访问的文件夹的名称
        in_filelist=dir(rootpath);           % 访问rootpath中的文件形成结构体，主要是获得二级目录里面的文件，放在filelist这个变量里面
        %ni = 12;
        ni=length(in_filelist);     % in_filelist里面到底有多少个元素
        for j=1:ni                 %依次循环遍历ni中的各个元素
            if strcmp(in_filelist(j).name,'.')|| strcmp(in_filelist(j).name,'..')|| strcmp(in_filelist(j).name,'Desktop_1.ini')|| strcmp(in_filelist(j).name,'Desktop_2.ini')  %设置跳出循环的条件
            else
                tempind=tempind+1;
                imglist{tempind}=imread(strcat(rootpath,'/',in_filelist(j).name));  %将读到的每个图片都写到imglist里面去imread函数读到的是一个三维矩阵数字是三维矩阵的RGB值，imglist是一个细胞矩阵
            end
        end
    end
end
end

