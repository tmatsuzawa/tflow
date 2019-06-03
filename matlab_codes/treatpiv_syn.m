%% PIV caller

function treatpiv_syn(Dirbase, dirlist, Dt, W, N, step, imheader, ext)

subratio = 10;
Data_name = ['/PIV_W' num2str(W) '_step' num2str(step) '_data'];
% Number of passes is 4 as a default. If not, specify below.
% % Number of passes
% if (W==8)
%     N=4;
% elseif (W==16)
%     N=3; 
% elseif (W==32)
%     N=2;
% elseif (W==64)
%     N=1;
% else
%     disp('No valid box size given')
% end

disp(length(dirlist))

for ind = 1:1:length(dirlist)
    directory=dirlist{ind};
    files= dir(['./' directory '/' imheader '*.' ext]);
    
    disp('Found images...')
    
    filenames={files.name};
%    filenames = sortrows(filenames); %sort all image files
    filenames = sort_nat(filenames); % sort in a natural order to humans
    amount = length(filenames);
    if rem(amount, 2) == 1
       amount = amount - 1;
    end
    
    %% Standard PIV Settings
    s = cell(10,2); % To make it more readable, let's create a "settings table"
    %Parameter                       %Setting           %Options
    s{1,1}= 'Int._area_1';           s{1,2}=W*2^3;         % window size of first pass
    s{2,1}= 'Step_size_1';           s{2,2}=W*2^2;         % step of first pass
    s{3,1}= 'Subpix._finder';        s{3,2}=2;          % 1 = 3point Gauss, 2 = 2D Gauss
    s{4,1}= 'Mask';                  s{4,2}=[];         % If needed, generate via: imagesc(image); [temp,Mask{1,1},Mask{1,2}]=roipoly;
    s{5,1}= 'ROI';                   s{5,2}=[];         % Region of interest: [x,y,width,height] in pixels, may be left empty
    s{6,1}= 'Nr._of_passes';         s{6,2}=N;          % 1-4 nr. of passes
    s{7,1}= 'Int._area_2';           s{7,2}=W*2^2;         % second pass window size
    s{8,1}= 'Int._area_3';           s{8,2}=W*2;         % third pass window size
    s{9,1}= 'Int._area_4';           s{9,2}=W;         % fourth pass window size
    s{10,1}='Window_deformation';    s{10,2}='spline'; % '*spline' is more accurate, but slower
    
    % Standard image preprocessing settings
    p = cell(8,1);
    %Parameter                       %Setting           %Options
    p{1,1}= 'ROI';                   p{1,2}=s{5,2};     % same as in PIV settings
    p{2,1}= 'CLAHE';                 p{2,2}=1;          % 1 = enable CLAHE (contrast enhancement), 0 = disable
    p{3,1}= 'CLAHE_size';            p{3,2}=10;         % CLAHE window size
    p{4,1}= 'Highpass';              p{4,2}=1;          % 1 = enable highpass, 0 = disable
    p{5,1}= 'Highpass_size';         p{5,2}=15;         % highpass size
    p{6,1}= 'Clipping';              p{6,2}=1;          % 1 = enable clipping, 0 = disable
    p{7,1}= 'Clipping_thresh.';      p{7,2}=1;          % 1 = enable wiener noise removing, 0 = disable
    p{8,1}= 'Intensity_Capping';     p{8,2}=3;          % 0-255 wiener parameter
    
    % PIV postprocessing settings
    %Parameter                       %Setting        %Descriptions
    post_set = cell(7,1);
    post_set{1,1} = 'umin';     post_set{1,2} = -500; % minimum allowed u velocity default:-2
    post_set{2,1} = 'umax';     post_set{2,2} =  500; % maximum allowed u velocity default:2
    post_set{3,1} = 'vmin';     post_set{3,2} = -500; % maximum allowed v velocity default:-2
    post_set{4,1} = 'vmax';     post_set{4,2} =  500; % maximum allowed v velocity default:2
    post_set{5,1} = 'stdthresh';post_set{5,2} =  4; % threshold for standard deviation check
    post_set{6,1} = 'epsilon';  post_set{6,2} = 0.15; % epsilon for normalized median test
    post_set{7,1} = 'thresh';   post_set{7,2} = 1;  % threshold for normalized median test
    
    
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
    if s{5+s{6,2},2} == 1
        PathName =[Dirbase Data_name '/PIVlab_ratio2_W64pix_Dt_' int2str(Dt) '_' basename];
    else 
        PathName =[Dirbase Data_name '/PIVlab_ratio2_W' int2str(s{5+s{6,2},2}) 'pix_Dt_' int2str(Dt) '_' basename];
    end
    mkdir(PathName);
    

    for i=1:step:(amount-Dt)
        %file_conv = [Dir 'Corr_map_128pix_Dt' num2str(Dt)];
        %write result in a txt file
        
        FileName = ['D' num2str(i, '%05.f') '.txt'];
        if exist(fullfile(PathName,FileName))~=2
            if i == 1
                % Save processing settings
                paramdir = [PathName '/piv_settings'];
                mkdir(paramdir);
                save_PIVlab_settings(p, s, post_set, paramdir, '/piv_settings.txt');
            end
            %  disp(i+1)
            counter=counter+1;
            disp(filenames{i})
            disp(filenames{i+1})
            image1 = imread(fullfile([Dirbase, directory], filenames{i})); % read imageA
            image2 = imread(fullfile([Dirbase, directory], filenames{i+Dt}));% read imageB
            image1 = PIVlab_preproc (image1,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2}); %preprocess images
            image2 = PIVlab_preproc (image2,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2});
            [x{1},y{1},u{1},v{1},typevector{1}] = piv_FFTmulti (image1,image2,s{1,2},s{2,2},s{3,2},s{4,2},s{5,2},s{6,2},s{7,2},s{8,2},s{9,2},s{10,2});%,file_conv);
                                                  %piv_FFTmulti (image1,image2,interrogationarea, step, subpixfinder, mask_inpt, roi_inpt,passes,int2,int3,int4,imdeform)
            
            typemask{1} = logical(not(isnan(u{1}))+not(isnan(v{1})));
            
            clc
            disp(['PIV all fields:' int2str(i/(amount-1)*100) ' %']); % displays the progress in command window
    
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
                u_filtered=inpaint_nans(u_filtered,4);
                v_filtered=inpaint_nans(v_filtered,4);

                u_filt{PIVresult,1}=u_filtered;
                v_filt{PIVresult,1}=v_filtered;
                typevector_filt{PIVresult,1}=typevector_filtered;
            end
            % save the data in a .txt file
            save_single_data_PIVlab(i,Dt,PathName,FileName,filenames,true,x{1},y{1},u_filt{1},v_filt{1}); %filtered vel field
            %save_single_data_PIVlab(i,Dt,PathName,FileName,filenames,true,x{1},y{1},u{1},v{1}); % unfiltered vel field
        else
            if (i==1)
                disp('File already exists, skip')
            end
        end
    end
    disp('Done')
end