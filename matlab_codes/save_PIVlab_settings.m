function save_PIVlab_settings(pre_set, std_set, post_set, dirbase, filename)
header = {};
header{1} = '--------------------------------';
header{2} = ['Time the data started to be processed: ' datestr(datetime())];
delimiter = '\t';

filepath = fullfile(dirbase,filename);
filepath2 = fullfile(dirbase, [filename(1:end-4) '_2.txt']);

params = transpose([pre_set; std_set; post_set]); % cell array
params_2 = [pre_set; std_set; post_set]; % for a better visibility to humans
dlmcell(filepath, params, header, delimiter, '-a')
dlmcell(filepath2, params_2, header, delimiter, '-a')