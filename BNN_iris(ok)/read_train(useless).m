function imglist = read_train(root)
% niΪ��ȡͼƬ������nΪ�ļ�����Ŀ
%========��ȡ�ļ���========%
out_Files = dir(root);%չ��
tempind=0;              % ��ʼ��һ��temibd����Ӧ������������֮���
imglist=cell(0);        %��ʼ��һ���յ�cell����
n=length(out_Files);    %Ӧ����out_Files������Ԫ�صĸ���
%========��ȡ�ļ�========%
for i = 1:n;            % �ӵ�һ��Ԫ�ؿ�ʼһ�����ķ���ÿ��Ԫ�أ�Ҳ���ǽṹ�壩
    if strcmp(out_Files(i).name,'.')|| strcmp(out_Files(i).name,'..')  % �ṹ�������Ϊ��.��,���ߡ�..��������forѭ�����������forѭ��
    else
        rootpath=strcat(root,'/',out_Files(i).name);  %���Ӻ����������ַ��������������������ǵõ���Ҫ���ʵ��ļ��е�����
        in_filelist=dir(rootpath);           % ����rootpath�е��ļ��γɽṹ�壬��Ҫ�ǻ�ö���Ŀ¼������ļ�������filelist�����������
        %ni = 12;
        ni=length(in_filelist);     % in_filelist���浽���ж��ٸ�Ԫ��
        for j=1:ni                 %����ѭ������ni�еĸ���Ԫ��
            if strcmp(in_filelist(j).name,'.')|| strcmp(in_filelist(j).name,'..')|| strcmp(in_filelist(j).name,'Desktop_1.ini')|| strcmp(in_filelist(j).name,'Desktop_2.ini')  %��������ѭ��������
            else
                tempind=tempind+1;
                imglist{tempind}=imread(strcat(rootpath,'/',in_filelist(j).name));  %��������ÿ��ͼƬ��д��imglist����ȥimread������������һ����ά������������ά�����RGBֵ��imglist��һ��ϸ������
            end
        end
    end
end
end

