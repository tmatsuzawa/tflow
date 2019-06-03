%% PIV caller

function treatpiv_s(Dirbase,dirlist,Dt,W,step,test)

subratio = 10;
Data_name = ['/PIV_W' num2str(W) '_step' num2str(step/2) '_data'];


if (test)
    Dir = Dirbase;
else
    Dir = [Dirbase '/Tiff_folder'];
end

if (W==16)
    N=4;
elseif (W==32)
    N=3;
elseif (W==64)
    N=2;
else
    disp('No valid box size given')
end

disp(length(dirlist))


for ind = 1:1:length(dirlist)
    directory=dirlist{ind};
    files= dir([Dir '/' directory '/*.tiff']);
    
    disp('Files index loaded')
    filenames={files.name};
    filenames = sortrows(filenames); %sort all image files
    amount = length(filenames);
    % Standard PIV Settings
    s = cell(10,2); % To make it more readable, let's create a "settings table"
    %Parameter                       %Setting           %Options
    s{1,1}= 'Int. area 1';           s{1,2}=128;         % window size of first pass
    s{2,1}= 'Step size 1';           s{2,2}=64;         % step of first pass
    s{3,1}= 'Subpix. finder';        s{3,2}=2;          % 1 = 3point Gauss, 2 = 2D Gauss
    s{4,1}= 'Mask';                  s{4,2}=[];         % If needed, generate via: imagesc(image); [temp,Mask{1,1},Mask{1,2}]=roipoly;
    s{5,1}= 'ROI';                   s{5,2}=[];         % Region of interest: [x,y,width,height] in pixels, may be left empty
    s{6,1}= 'Nr. of passes';         s{6,2}=N;          % 1-4 nr. of passes
    s{7,1}= 'Int. area 2';           s{7,2}=64;         % second pass window size
    s{8,1}= 'Int. area 3';           s{8,2}=32;         % third pass window size
    s{9,1}= 'Int. area 4';           s{9,2}=16;         % fourth pass window size
    s{10,1}='Window deformation';    s{10,2}='spline'; % '*spline' is more accurate, but slower
    
    % Standard image preprocessing settings
    p = cell(8,1);
    %Parameter                       %Setting           %Options
    p{1,1}= 'ROI';                   p{1,2}=s{5,2};     % same as in PIV settings
    p{2,1}= 'CLAHE';                 p{2,2}=1;          % 1 = enable CLAHE (contrast enhancement), 0 = disable
    p{3,1}= 'CLAHE size';            p{3,2}=10;         % CLAHE window size
    p{4,1}= 'Highpass';              p{4,2}=1;          % 1 = enable highpass, 0 = disable
    p{5,1}= 'Highpass size';         p{5,2}=15;         % highpass size
    p{6,1}= 'Clipping';              p{6,2}=1;          % 1 = enable clipping, 0 = disable
    p{7,1}= 'Clipping thresh.';      p{7,2}=1;          % 1 = enable wiener noise removing, 0 = disable
    p{8,1}= 'Intensity Capping';     p{8,2}=3;          % 0-255 wiener parameter
    
    
    % PIV analysis loop
    x=cell(1);
    y=x;
    u=x;
    v=x;
    typevector=x; %typevector will be 1 for regular vectors, 0 for masked areas
    typemask=x;
    counter=0;
    
    %create the directory to save the data :
    basename = directory(1:end-5); %remove the _File extension
    PathName =[Dirbase Data_name '/PIVlab_ratio2_W' int2str(s{5+s{6,2},2}) 'pix_Dt_' int2str(Dt) '_' basename];
    mkdir(PathName);
    
%    if test 
%        amount=Dt+2;
%    end
    for i=1:step:(amount-Dt)
        %file_conv = [Dir 'Corr_map_128pix_Dt' num2str(Dt)];
        %write result in a txt file
        ndigit = floor(log10(amount*subratio))+1;
        number = str2num(filenames{i}(3:7));        
        disp(number)
        if number>0
            nzero = ndigit -(floor(log10(number))+1);
        else
            nzero = ndigit -(floor(log10(1))+1)+1;
        end
        
        zeros='';
        for k=1:nzero
            zeros=[zeros '0'];
        end
        FileName = ['D' zeros int2str(number) '.txt'];
        
        %   disp(FileName)
        disp(fullfile(PathName,FileName))
        if exist(fullfile(PathName,FileName))~=2
            %  disp(i+1)
            counter=counter+1;
            image1 = imread(fullfile([Dir '/' directory], filenames{i})); % read images
            image2 = imread(fullfile([Dir '/' directory], filenames{i+Dt}));
            image1 = PIVlab_preproc (image1,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2}); %preprocess images
            image2 = PIVlab_preproc (image2,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2});
            [x{1},y{1},u{1},v{1},typevector{1}] = piv_FFTmulti (image1,image2,s{1,2},s{2,2},s{3,2},s{4,2},s{5,2},s{6,2},s{7,2},s{8,2},s{9,2},s{10,2});%,file_conv);
                                                  %piv_FFTmulti (image1,image2,interrogationarea, step, subpixfinder, mask_inpt, roi_inpt,passes,int2,int3,int4,imdeform)
            
            typemask{1} = logical(not(isnan(u{1}))+not(isnan(v{1})));
            
            clc
            disp(['PIV all fields:' int2str(i/(amount-1)*100) ' %']); % displays the progress in command window
         
            %temp_varname = find(directory == '/');
            %varname = directory((temp_varname(end-1)+1):(temp_varname(end)-1));
            % clearvars -except p s x y u v typevector xper yper uper vper typevectorper directory filenames u_filt v_filt typevector_filt dmoy typemask typemaskper varname dirlist ind
            
            save_single_data_PIVlab(i,Dt,PathName,FileName,filenames,true,x{1},y{1},u{1},v{1})
        else
            if (i==1)
                disp(fullfile(PathName,FileName))
                disp('File already exists, skip')
            end
        end
        %  pause
        %        cd(directory);
        %    name = ['ratio2_W32pix_' basename '_'];
        %    save([name date]);
        %    cd('..');
        %     % Graphical output (disable to improve speed)
        %     %%{
        %     imagesc(double(image1)+double(image2));colormap('gray');
        %     hold on
        %     quiver(x{counter},y{counter},u{counter},v{counter},'g','AutoScaleFactor', 1.5);
        %     hold off;
        %     axis image;
        %     title(filenames{i},'interpreter','none')
        %     set(gca,'xtick',[],'ytick',[])
        %     drawnow;
        %     %%}
    end
    disp('Done')
end