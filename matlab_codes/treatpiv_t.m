%% PIV caller

function treatpiv_t(Dirbase, dirlist, Dt, W, step, maxind, bitnumber, reprocess)

subratio = 1;
Data_name = ['/pivlab_outputs/PIV_W' num2str(W) '_Dt' num2str(Dt) '_step' num2str(step) '_data'];

% Number of passes
if (W==8)
    N=4;
elseif (W==16)
    N=3; 
elseif (W==32)
    N=2;
elseif (W==64)
    N=1;
else
    disp('No valid box size given')
end

% disp(length(dirlist));

for ind = 1:1:length(dirlist)
    directory=dirlist{ind};
    files= dir(['./' directory '/im*.tiff']);
%     files= dir(['./' directory '/npts0050000_shape01024x01024_z50_lthickness20p00_nsteps010_zeta1p00_sz1p500_gaussian_szsig0p500_shapegaussian_lsig0p100_zplane_lsig0p100_zplane_cutoff1p000_maxi255p000_pbcTrue_*.png']);
    disp(['Found ' int2str(length(files)) ' images...'])
    amount = length(files);
    filenames={files.name};
%     filenames = sortrows(filenames); %sort all image files
    filenames = sort_nat(filenames);
    amount = length(filenames);
    % if number of image is odd, it will raise an error. 
    if rem(amount, 2) == 1
        amount = amount-1;
    end
%     disp(filenames)
    
    % process images till max
    if maxind == 0  % Default: process all
        maxind = amount - Dt;
    end

    %% Standard PIV Settings
    std_set = cell(12,2); % To make it more readable, let's create a "settings table"
    %Parameter                       %Setting           %Options
    std_set{1,1}= 'Int._area_1';           std_set{1,2}=64;         % window size of first pass
    std_set{2,1}= 'Step_size_1';           std_set{2,2}=32;         % step of first pass
    std_set{3,1}= 'Subpix._finder';        std_set{3,2}=2;          % 1 = 3point Gauss, 2 = 2D Gauss
    std_set{4,1}= 'Mask';                  std_set{4,2}=[];         % If needed, generate via: imagesc(image); [temp,Mask{1,1},Mask{1,2}]=roipoly;
    std_set{5,1}= 'ROI';                   std_set{5,2}=[];         % Region of interest: [x,y,width,height] in pixels, may be left empty
    std_set{6,1}= 'Nr._of_passes';         std_set{6,2}=N;          % 1-4 nr. of passes
    std_set{7,1}= 'Int._area_2';           std_set{7,2}=32;         % second pass window size
    std_set{8,1}= 'Int._area_3';           std_set{8,2}=16;         % third pass window size
    std_set{9,1}= 'Int._area_4';           std_set{9,2}=8;          % fourth pass window size
    std_set{10,1}='Window_deformation';    std_set{10,2}='*spline';  % '*spline' is more accurate, but slower
    std_set{11,1}='Dt';                    std_set{11,2}=Dt;        % spacing between imageA and imageB
    std_set{12,1}='step';                  std_set{12,2}=step;      % spacing between successive image pairs
    std_set{13,1}='repeat';                std_set{12,2}=0;      % if 1, repeat computing correlation
    std_set{14,1}='mask_auto';             std_set{12,2}=1;      % if 1, repalce the center of matrix (3x3) by its mean
    
    % Standard image preprocessing settings
    pre_set = cell(8,1);
    %Parameter                       %Setting           %Options
    pre_set{1,1}= 'ROI';                   pre_set{1,2}=std_set{5,2};     % same as in PIV settings
    pre_set{2,1}= 'CLAHE';                 pre_set{2,2}=1;          % 1 = enable CLAHE (contrast enhancement), 0 = disable
    pre_set{3,1}= 'CLAHE_size';            pre_set{3,2}=10;         % CLAHE window size
    pre_set{4,1}= 'Highpass';              pre_set{4,2}=1;          % 1 = enable highpass, 0 = disable
    pre_set{5,1}= 'Highpass_size';         pre_set{5,2}=15;         % highpass size
    pre_set{6,1}= 'Clipping';              pre_set{6,2}=1;          % 1 = enable clipping, 0 = disable
    pre_set{7,1}= 'Clipping_thresh.';      pre_set{7,2}=1;          % 1 = enable wiener noise removing, 0 = disable
    pre_set{8,1}= 'Intensity_Capping';     pre_set{8,2}=3;          % 0-255 wiener parameter
    
    % PIV postprocessing settings
    % Parameter                       %Setting        %Descriptions
    post_set = cell(7,1);
    post_set{1,1} = 'umin';     post_set{1,2} = -2; % minimum allowed u velocity
    post_set{2,1} = 'umax';     post_set{2,2} =  2; % maximum allowed u velocity
    post_set{3,1} = 'vmin';     post_set{3,2} = -2; % maximum allowed v velocity
    post_set{4,1} = 'vmax';     post_set{4,2} =  2; % maximum allowed v velocity
    post_set{5,1} = 'stdthresh';post_set{5,2} =  2; % threshold for standard deviation check
    post_set{6,1} = 'epsilon';  post_set{6,2} = 0.15; % epsilon for normalized median test
    post_set{7,1} = 'thresh';   post_set{7,2} = 2;  % threshold for normalized median test
    
    
    % PIV analysis loop
    x=cell(1);
    y=x;
    u=x;
    v=x;
    typevector=x; %typevector will be 1 for regular vectors, 0 for masked areas
    typemask=x;
    counter=0;
    
    %create the directory to save the data :
    basename = directory; %remove the _File extension
    PathName =[Dirbase Data_name '/PIVlab_Win' int2str(std_set{1,2}) 'pix_W' int2str(W) 'px_Dt' int2str(Dt) '_step' num2str(step) '_' basename];
    if exist(PathName) && reprocess
        datadirs = dir([Dirbase Data_name '/PIVlab_*']);
        n_datadirs = size(datadirs,1);
        PathName = [PathName '_' sprintf('%02d', n_datadirs)];
    end
        
    mkdir(PathName);
    
   
    for i=1:step:(maxind)
        %write result in a txt file
        ndigit = floor(log10(amount*subratio))+1;
%         number = str2num(filenames{i}(3:8));% for im00000.tiff
        number = i;
%         disp(number)
        if number>0
            nzero = ndigit -(floor(log10(number))+1);
        else
            nzero = ndigit -(floor(log10(1))+1)+1;
        end
        
        index = sprintf('%06d', number);
        
        
        FileName = ['D' index '.txt'];
        
        %   disp(PathName, FileName)
%         disp(fullfile(PathName,FileName))
        if exist(fullfile(PathName,FileName))~=2
            if i == 1
                % Save processing settings
                paramdir = [PathName '/piv_settings'];
                mkdir(paramdir)
                save_PIVlab_settings(pre_set, std_set, post_set, paramdir, '/piv_settings.txt')
            end
            %  disp(i+1)
            counter=counter+1;
            disp(i)
            disp(fullfile(['./' directory], filenames{i}))
            image1 = imread(fullfile(['./' directory], filenames{i})); % read images
            image2 = imread(fullfile(['./' directory], filenames{i+Dt}));
            minintens_1 = min(image1(:));
            maxintens_1 = max(image1(:));
            minintens_2 = min(image2(:));
            maxintens_2 = max(image2(:));
%             minintens = min([minintens_1 minintens_2])  / (2^bitnumber);
%             maxintens = max([maxintens_1 maxintens_2])  / (2^bitnumber);
            image1 = PIVlab_preproc (image1,pre_set{1,2},pre_set{2,2},pre_set{3,2},pre_set{4,2},pre_set{5,2},pre_set{6,2},pre_set{7,2},pre_set{8,2}); %preprocess images
            image2 = PIVlab_preproc (image2,pre_set{1,2},pre_set{2,2},pre_set{3,2},pre_set{4,2},pre_set{5,2},pre_set{6,2},pre_set{7,2},pre_set{8,2});
            [x{1},y{1},u{1},v{1},typevector{1}] = piv_FFTmulti (image1,image2,std_set{1,2},std_set{2,2},std_set{3,2},std_set{4,2},std_set{5,2},std_set{6,2},std_set{7,2},std_set{8,2},std_set{9,2},std_set{10,2});%,file_conv);
%             % For new release of PIVLab
%             image1 = PIVlab_preproc (image1,pre_set{1,2},pre_set{2,2},pre_set{3,2},pre_set{4,2},pre_set{5,2},pre_set{6,2},pre_set{7,2},pre_set{8,2}, minintens, maxintens); %preprocess images
%             image2 = PIVlab_preproc (image2,pre_set{1,2},pre_set{2,2},pre_set{3,2},pre_set{4,2},pre_set{5,2},pre_set{6,2},pre_set{7,2},pre_set{8,2}, minintens, maxintens);
%             [x{1},y{1},u{1},v{1},typevector{1}] = piv_FFTmulti (image1,image2,std_set{1,2},std_set{2,2},std_set{3,2},std_set{4,2},std_set{5,2},std_set{6,2},std_set{7,2},std_set{8,2},std_set{9,2},std_set{10,2},std_set{13,2},std_set{14,2});%,file_conv);
            
            %piv_FFTmulti (image1,image2,interrogationarea, step, subpixfinder, mask_inpt, roi_inpt,passes,int2,int3,int4,imdeform)
            
            typemask{1} = logical(not(isnan(u{1}))+not(isnan(v{1})));
            
            clc
            disp(['PIV all fields:' int2str(i/(amount-1)*100) ' %']) % displays the progress in command window
    
            % PIV postprocessing loop
            umin = post_set{1,2};
            umax = post_set{2,2};
            vmin = post_set{3,2};
            vmax = post_set{4,2};
            stdthresh = post_set{5,2};
            epsilon = post_set{6,2};
            thresh = post_set{7,2};

            u_filt=cell(amount/2,1);
            v_filt=u_filt;
            typevector_filt=u_filt;
            for PIVresult=1:size(x,1)
                u_filtered=u{PIVresult,1};
                v_filtered=v{PIVresult,1};
                typevector_filtered=typevector{PIVresult,1};
                %vellimit check
                u_filtered(u_filtered<umin)=NaN;
                u_filtered(u_filtered>umax)=NaN;
                v_filtered(v_filtered<vmin)=NaN;
                v_filtered(v_filtered>vmax)=NaN;
                % stddev check
                meanu=nanmean(nanmean(u_filtered));
                meanv=nanmean(nanmean(v_filtered));
                std2u=nanstd(reshape(u_filtered,size(u_filtered,1)*size(u_filtered,2),1));
                std2v=nanstd(reshape(v_filtered,size(v_filtered,1)*size(v_filtered,2),1));
                minvalu=meanu-stdthresh*std2u;
                maxvalu=meanu+stdthresh*std2u;
                minvalv=meanv-stdthresh*std2v;
                maxvalv=meanv+stdthresh*std2v;
                u_filtered(u_filtered<minvalu)=NaN;
                u_filtered(u_filtered>maxvalu)=NaN;
                v_filtered(v_filtered<minvalv)=NaN;
                v_filtered(v_filtered>maxvalv)=NaN;
                
                % normalized median check
                %Westerweel & Scarano (2005): Universal Outlier detection for PIV data
                [J,I]=size(u_filtered);
                %a=zeros(J,I)
                medianres=zeros(J,I);
                normfluct=zeros(J,I,2);
                b=1;
                for c=1:2
                    if c==1; velcomp=u_filtered;else;velcomp=v_filtered;end %#ok<*NOSEM>
                    for ii=1+b:I-b
                        for j=1+b:J-b
                            neigh=velcomp(j-b:j+b,ii-b:ii+b);
                            neighcol=neigh(:);
                            neighcol2=[neighcol(1:(2*b+1)*b+b);neighcol((2*b+1)*b+b+2:end)];
                            med=median(neighcol2);
                            fluct=velcomp(j,ii)-med;
                            res=neighcol2-med;
                            medianres=median(abs(res));
                            normfluct(j,ii,c)=abs(fluct/(medianres+epsilon));
                        end
                    end
                end
                info1=(sqrt(normfluct(:,:,1).^2+normfluct(:,:,2).^2)>thresh);
                u_filtered(info1==1)=NaN;
                v_filtered(info1==1)=NaN;

                typevector_filtered(isnan(u_filtered))=2;
                typevector_filtered(isnan(v_filtered))=2;
                typevector_filtered(typevector{PIVresult,1}==0)=0; %restores typevector for mask

                %Interpolate missing data
                u_filtered=inpaint_nans(u_filtered, 4);
                v_filtered=inpaint_nans(v_filtered, 4);

                u_filt{PIVresult,1}=u_filtered;
                v_filt{PIVresult,1}=v_filtered;
                typevector_filt{PIVresult,1}=typevector_filtered;
            end
            % save the data in a .txt file
            save_single_data_PIVlab(i,Dt,PathName,FileName,filenames,true,x{1},y{1},u_filt{1},v_filt{1}); %filtered vel field
            %save_single_data_PIVlab(i,Dt,PathName,FileName,filenames,true,x{1},y{1},u{1},v{1}) % unfiltered vel field
        else
            if (i==1)
                disp('File already exists, skip')
            end
        end
    end
    
   % reset max to process all for the next file
    if maxind == amount - Dt  % Default: process all
        maxind = 0;
    end
    
    disp('Done')
end