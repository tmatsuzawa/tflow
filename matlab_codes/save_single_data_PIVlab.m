function save_single_data_PIVlab(currentframe,Dt,PathName,FileName,filenames,export_vort,x,y,u,v)
%%%%%%%
% Function to save a PIVLab outputs
%%%%%%%
delimiter = ',';
calxy = 1;
caluv = 1;

header = {};
% disp([currentframe,Dt,PathName,FileName,filenames,export_vort,x,y,u,v]);
header{1}=['PIVlab by W.Th. & E.J.S., ASCII chart output - ' date];
header{2}=['FRAME: ' int2str(currentframe) ', filenames: A: ' filenames{currentframe} ' & B: ' filenames{currentframe+Dt} ', conversion factor xy (px -> m): ' num2str(calxy) ', conversion factor uv (px/frame -> m/s): ' num2str(caluv)];
%get(handles.export_vort, 'Value')
if export_vort
    header{3}=['x [px]' delimiter 'y [px]' delimiter 'u [px/frame]' delimiter 'v [px/frame]' delimiter 'vorticity [1/frame]'];%delimiter 'magnitude[m/s]' delimiter 'divergence[1]' delimiter 'vorticity[1/s]' delimiter 'dcev[1]']
else
    header{3}=['x [px]' delimiter 'y [px]' delimiter 'u [px/frame]' delimiter 'v [px/frame]'];%delimiter 'magnitude[m/s]' delimiter 'divergence[1]' delimiter 'vorticity[1/s]' delimiter 'dcev[1]']
end

disp(['Saving ' fullfile(PathName,FileName)])
fid = fopen(fullfile(PathName,FileName), 'w');
for j=1:length(header)
    fprintf(fid, [header{j} '\r\n']);
end
fclose(fid);

subtract_u = 0;
subtract_v = 0;

% Export vorticity data if selected
if export_vort
    [vort, ~] = curl(x,y,u,v);
    wholeLOT=[reshape(x*calxy,size(x,1)*size(x,2),1) reshape(y*calxy,size(y,1)*size(y,2),1) reshape(u*caluv-subtract_u,size(u,1)*size(u,2),1) reshape(v*caluv-subtract_v,size(v,1)*size(v,2),1) reshape(vort,size(vort,1)*size(vort,2),1)];
else
    wholeLOT=[reshape(x*calxy,size(x,1)*size(x,2),1) reshape(y*calxy,size(y,1)*size(y,2),1) reshape(u*caluv-subtract_u,size(u,1)*size(u,2),1) reshape(v*caluv-subtract_v,size(v,1)*size(v,2),1)];
end

dlmwrite(fullfile(PathName,FileName), wholeLOT, '-append', 'delimiter', delimiter, 'precision', 10, 'newline', 'pc');