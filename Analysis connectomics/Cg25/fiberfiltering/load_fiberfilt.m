addpath('C:\code\wjn_toolbox');
addpath(genpath('C:\code\leaddbs'));
addpath(genpath('C:\code\spm12'));


% created with fiberfiltering.mat
load('Cg25_BalancedAccuracy_2.fibfilt','-mat')

groups = []; 
groups_lochout = [];
cnt = 1;
for i=1:8
    for j=1:8
        groups = [groups i];
        groups_lochout = [groups_lochout cnt]; 
        cnt = cnt + 1;
    end
end

[I, Ihat] = tractset.crossval(groups);

resultfig = ea_mnifigure;

ea_discfiberexplorer('Cg25_BalancedAccuracy_2.fibfilt',resultfig)

