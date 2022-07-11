PATH_OUT = "C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics\new_conn_out\str_conn_smooth_888";
PATH_IN = "C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics\new_conn_out\str_conn";

addpath('C:\code\wjn_toolbox');
addpath(genpath('C:\code\leaddbs'));
addpath(genpath('C:\code\spm12'));

nii_files = dir("C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\scripts\Analysis connectomics\new_conn_out\str_conn\*nii");

for i=1:size(nii_files, 1)
    in_ = fullfile(PATH_IN, string(nii_files(i).name));
    out_ = fullfile(PATH_OUT, strcat("s", string(nii_files(i).name)));
    in_ = convertStringsToChars(in_);
    out_ = convertStringsToChars(out_);
    spm_smooth(in_, out_, [8 8 8]);
end