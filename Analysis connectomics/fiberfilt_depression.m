addpath('C:\Users\ICN_admin\Documents\wjn_toolbox');
addpath(genpath('C:\Users\ICN_admin\Documents\leaddbs'));
addpath(genpath('C:\Users\ICN_admin\Documents\spm12'));

csvfile = "electrodes.csv";


T=readtable(csvfile);


mni_coords = [T.x T.y T.z];
mkdir Cg25
cd('Cg25')
fname = 'Cg25_BalancedAccuracy';

% Create region of interest nifti files
cg25_files={};cg25_group = [];roi_radius = 10;
group = 1;
for a=1:size(mni_coords,1)
    f_ch_name = convertStringsToChars(strcat(string(a), '_sub-', string(T.sub(a)), '_', T.ch{a}, '.nii'));
    wjn_spherical_roi( ...
        f_ch_name,...
        mni_coords(a,:), ...
        roi_radius, ...
        fullfile(spm('dir'),'canonical','avg152T1.nii'));
    cg25_files{a,1}=f_ch_name;
    if mod(a, 8) == 0
        group = group + 1;
    end
    cg25_group(a) = group;
    %cg25_group(a) = T.sub(a);
end

% Create a Pseudo Lead-Group structure
M.pseudoM = 1; % Declare this is a pseudo-M struct, i.e. not a real lead group file
M.ROI.list=cg25_files; % enter the new files creates from MNI coordinates here
M.ROI.group=ones(length(cg25_files),1);

M.ROI.group=cg25_group';

M.clinical.labels={'per_balanced_acc','per_balanced_acc_norm'}; % how will variables be called
M.clinical.vars{1}=T.per; % enter a variable of interest - entries correspond to nifti files
M.clinical.vars{2}=wjn_gaussianize(T.per);
M.guid=fname; % give your analysis a name
save(fname,'M'); % store data of analysis to file

resultfig = ea_mnifigure; % Create empty 3D viewer figure

% Open up the Fiber Filtering Explorer
ea_discfiberexplorer(fullfile(pwd,fname),resultfig);

