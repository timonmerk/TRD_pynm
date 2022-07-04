addpath('C:\Users\ICN_admin\Documents\wjn_toolbox');
addpath(genpath('C:\Users\ICN_admin\Documents\leaddbs'));
addpath(genpath('C:\Users\ICN_admin\Documents\spm12'));

spm('defaults','eeg');

f_names = dir('*.dat');

for i = 1:numel(f_names)
    f_name = f_names(i).name(1:end-4);
    f_name = 'effspm8_KSC_EMO'
    
    dat = spm_eeg_load(strcat(f_name, '.mat'));
    label = load(strcat(f_name, '.mat'));
    
    label_val = [];
    for j = 1:numel(label.D.trials)
        label_val = [label_val, label.D.trials(j).events.value];
    end
    
    %dat(end+1, 1, :) = reshape(label_val, 1, 1, size(label_val, 2));
    %dat_new = zeros(size(dat, 1)+1, size(dat,2), size(dat, 3));
    %dat_new(1:size(dat,1),:, :) = dat;
    %dat_new(end,:,:) = reshape(label_val, 1, 1, size(label_val, 2));
   
    D = struct;
    D.data = dat(:,:,:);
    D.ch_names = dat.chanlabels;
    D.labels = {label.D.trials.label};
    D.bad = {label.D.trials.bad};
    D.onset = {label.D.trials.onset};
    D.fsample = dat.fsample;
    
    save(strcat(f_name, '_edit.mat'), 'D');
end