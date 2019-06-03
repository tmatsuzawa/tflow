%% PIV caller

function treatpiv(Dirbase,dirlist,Dt,test)

if (test)
    Dir = Dirbase;
else
    Dir = [Dirbase '/Tiff_folder'];
end
for ind = 1:1:length(dirlist)
    %for ind = [8 9 10 12]
    %   clearvars -except ind dirlist
    % Create list of images inside specified directory
    %    directory=dirlist(ind).name; %directory containing the images you want to analyze
    %    suffix='*.tiff'; %*.bmp or *.tif or *.jpg
    %    direc = ls([directory,filesep,suffix]); filenames={};
    %   [filenames{1:length(direc),1}] = deal(direc.name);
    directory=dirlist{ind};
    if test
        files = dir([directory '/*.tiff']);
        disp([directory '/*.tiff'])
    else
        files= dir([Dir '/' directory '/*.tiff']);
        disp([Dir '/' directory '/*.tiff'])
    end
    
    disp(files)
    disp('Files index loaded')
    filenames={files.name};
    filenames = sortrows(filenames); %sort all image files
    amount = length(filenames);
    disp(amount)
    % Standard PIV Settings
    s = cell(10,2); % To make it more readable, let's create a "settings table"
    %Parameter                       %Setting           %Options
    s{1,1}= 'Int. area 1';           s{1,2}=128;         % window size of first pass
    s{2,1}= 'Step size 1';           s{2,2}=64;         % step of first pass
    s{3,1}= 'Subpix. finder';        s{3,2}=2;          % 1 = 3point Gauss, 2 = 2D Gauss
    s{4,1}= 'Mask';                  s{4,2}=[];         % If needed, generate via: imagesc(image); [temp,Mask{1,1},Mask{1,2}]=roipoly;
    s{5,1}= 'ROI';                   s{5,2}=[];         % Region of interest: [x,y,width,height] in pixels, may be left empty
    s{6,1}= 'Nr. of passes';         s{6,2}=2;          % 1-4 nr. of passes
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
    
    % if mod(amount,2) == 1 %Uneven number of images?
    %     disp('Image folder should contain an even number of images.')
    %     %remove last image from list
    %     amount=amount-1;
    %     filenames(size(filenames,1))=[];
    % end
    x=cell(amount-1,1);
    y=x;
    u=x;
    v=x;
    typevector=x; %typevector will be 1 for regular vectors, 0 for masked areas
    typemask=x;
    counter=0;
    
    disp(s{6,2})
    %create the directory to save the data :
    basename = directory(1:end-5); %remove the _File extension
    PathName =[Dirbase '/PIV_data' '/PIVlab_ratio20_W' int2str(s{5+s{6,2},2}) 'pix_Dt_' int2str(Dt) '_' basename];
    disp(PathName)
    mkdir(PathName)
    
    if test 
        amount=Dt+2;
    end
    for i=1:1:(amount-Dt)
        file_conv = [Dir 'Corr_map_128pix_Dt' num2str(Dt)];
        %write result in a txt file
        ndigit = floor(log10(amount))+1;
        nzero = ndigit -(floor(log10(i))+1);
        zeros='';
        for k=1:nzero
            zeros=[zeros '0'];
        end
        FileName = ['D' zeros int2str(i) '.txt'];
        %   disp(FileName)
        
        if exist(fullfile(PathName,FileName))~=2
            %  disp(i+1)
            counter=counter+1;
            image1=imread(fullfile(directory, filenames{i})); % read images
            image2=imread(fullfile(directory, filenames{i+Dt}));
            image1 = PIVlab_preproc (image1,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2}); %preprocess images
            image2 = PIVlab_preproc (image2,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2});
            [x{counter},y{counter},u{counter},v{counter},typevector{counter}] = piv_FFTmulti (image1,image2,s{1,2},s{2,2},s{3,2},s{4,2},s{5,2},s{6,2},s{7,2},s{8,2},s{9,2},s{10,2},file_conv);
            typemask{counter} = logical(not(isnan(u{counter}))+not(isnan(v{counter})));
            clc
            disp(directory);
            disp(['PIV all fields:' int2str(i/(amount-1)*100) ' %']);
            
            
            %temp_varname = find(directory == '/');
            %varname = directory((temp_varname(end-1)+1):(temp_varname(end)-1));
            % clearvars -except p s x y u v typevector xper yper uper vper typevectorper directory filenames u_filt v_filt typevector_filt dmoy typemask typemaskper varname dirlist ind
            
            save_single_data_PIVlab(i,Dt,PathName,FileName,filenames,true,x{counter},y{counter},u{counter},v{counter})
        else
            if (i==1)
                disp(fullfile(PathName,FileName))
                disp('File already exist, skip')
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
    
    % if mod(amount,2) == 1 %Uneven number of images?
    %     disp('Image folder should contain an even number of images.')
    %     %remove last image from list
    %     amount=amount-1;
    %     filenames(size(filenames,1))=[];
    % end
    if 0
        comp=0;
        for i=1:1:10
            xper{i}=cell(length(i:10:(amount-1)),1);
            yper{i}=xper{i};
            uper{i}=xper{i};
            vper{i}=xper{i};
            typevectorper{i}=xper{i}; %typevector will be 1 for regular vectors, 0 for masked areas
            typemaskper{i}=xper{i};
            countper=0;
            for j=i:10:(amount-10)
                comp=comp+1;
                countper=countper+1;
                image1=imread(fullfile(directory, filenames{j})); % read images
                image2=imread(fullfile(directory, filenames{j+10}));
                image1 = PIVlab_preproc (image1,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2}); %preprocess images
                image2 = PIVlab_preproc (image2,p{1,2},p{2,2},p{3,2},p{4,2},p{5,2},p{6,2},p{7,2},p{8,2});
                [xper{i}{countper} yper{i}{countper} uper{i}{countper} vper{i}{countper} typevectorper{i}{countper}] = piv_FFTmulti (image1,image2,s{1,2},s{2,2},s{3,2},s{4,2},s{5,2},s{6,2},s{7,2},s{8,2},s{9,2},s{10,2},file_conv);
                typemaskper{i}{countper} = logical(not(isnan(uper{i}{countper}))+not(isnan(vper{i}{countper})));
                clc
                disp(directory);
                disp(['PIV periodic fields:' int2str(comp/(amount-1)*100) ' %']);
                
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
        end
        
        % %% PIV postprocessing loop
        % % Settings
        % umin = -10; % minimum allowed u velocity
        % umax = 10; % maximum allowed u velocity
        % vmin = -10; % minimum allowed v velocity
        % vmax = 10; % maximum allowed v velocity
        % stdthresh=6; % threshold for standard deviation check
        % epsilon=0.15; % epsilon for normalized median test
        % thresh=3; % threshold for normalized median test
        %
        % u_filt=cell(amount/2,1);
        % v_filt=u_filt;
        % typevector_filt=u_filt;
        % for PIVresult=1:size(x,1)
        %     u_filtered=u{PIVresult,1};
        %     v_filtered=v{PIVresult,1};
        %     typevector_filtered=typevector{PIVresult,1};
        %     %vellimit check
        %     u_filtered(u_filtered<umin)=NaN;
        %     u_filtered(u_filtered>umax)=NaN;
        %     v_filtered(v_filtered<vmin)=NaN;
        %     v_filtered(v_filtered>vmax)=NaN;
        %     % stddev check
        %     meanu=nanmean(nanmean(u_filtered));
        %     meanv=nanmean(nanmean(v_filtered));
        %     std2u=nanstd(reshape(u_filtered,size(u_filtered,1)*size(u_filtered,2),1));
        %     std2v=nanstd(reshape(v_filtered,size(v_filtered,1)*size(v_filtered,2),1));
        %     minvalu=meanu-stdthresh*std2u;
        %     maxvalu=meanu+stdthresh*std2u;
        %     minvalv=meanv-stdthresh*std2v;
        %     maxvalv=meanv+stdthresh*std2v;
        %     u_filtered(u_filtered<minvalu)=NaN;
        %     u_filtered(u_filtered>maxvalu)=NaN;
        %     v_filtered(v_filtered<minvalv)=NaN;
        %     v_filtered(v_filtered>maxvalv)=NaN;
        %     % normalized median check
        %     %Westerweel & Scarano (2005): Universal Outlier detection for PIV data
        %     [J,I]=size(u_filtered);
        %     medianres=zeros(J,I);
        %     normfluct=zeros(J,I,2);
        %     b=1;
        %     for c=1:2
        %         if c==1; velcomp=u_filtered;else;velcomp=v_filtered;end %#ok<*NOSEM>
        %         for i=1+b:I-b
        %             for j=1+b:J-b
        %                 neigh=velcomp(j-b:j+b,i-b:i+b);
        %                 neighcol=neigh(:);
        %                 neighcol2=[neighcol(1:(2*b+1)*b+b);neighcol((2*b+1)*b+b+2:end)];
        %                 med=median(neighcol2);
        %                 fluct=velcomp(j,i)-med;
        %                 res=neighcol2-med;
        %                 medianres=median(abs(res));
        %                 normfluct(j,i,c)=abs(fluct/(medianres+epsilon));
        %             end
        %         end
        %     end
        %     info1=(sqrt(normfluct(:,:,1).^2+normfluct(:,:,2).^2)>thresh);
        %     u_filtered(info1==1)=NaN;
        %     v_filtered(info1==1)=NaN;
        %
        %     typevector_filtered(isnan(u_filtered))=2;
        %     typevector_filtered(isnan(v_filtered))=2;
        %     typevector_filtered(typevector{PIVresult,1}==0)=0; %restores typevector for mask
        %
        %     %Interpolate missing data
        %     u_filtered=inpaint_nans(u_filtered,4);
        %     v_filtered=inpaint_nans(v_filtered,4);
        %
        %     u_filt{PIVresult,1}=u_filtered;
        %     v_filt{PIVresult,1}=v_filtered;
        %     typevector_filt{PIVresult,1}=typevector_filtered;
        % end
        % clearvars -except p s x y u v typevector directory filenames u_filt v_filt typevector_filt
        
        % Averaging fields & saving data
        u
        v
        dmoy.uglob = zeros(size(u{1},1),size(u{1},2));
        dmoy.vglob = zeros(size(v{1},1),size(v{1},2));
        dmoy.uperglob = zeros(size(uper{1}{1},1),size(uper{1}{1},2));
        dmoy.vperglob = zeros(size(vper{1}{1},1),size(vper{1}{1},2));
        for i=1:10
            dmoy.umoy{i} = zeros(size(u{1},1),size(u{1},2));
            dmoy.vmoy{i} = zeros(size(v{1},1),size(v{1},2));
            for j=i:10:(amount-10)
                u{j}(isnan(u{j})) = 0;
                v{j}(isnan(v{j})) = 0;
                dmoy.umoy{i} = dmoy.umoy{i} + (1/length(i:10:(amount-10))).*u{j};
                dmoy.vmoy{i} = dmoy.vmoy{i} + (1/length(i:10:(amount-10))).*v{j};
            end;
            dmoy.uglob = dmoy.uglob + dmoy.umoy{i};
            dmoy.vglob = dmoy.vglob + dmoy.vmoy{i};
            
            dmoy.upermoy{i} = zeros(size(uper{1}{1},1),size(uper{1}{1},2));
            dmoy.vpermoy{i} = zeros(size(vper{1}{1},1),size(vper{1}{1},2));
            for j=1:1:length(i:10:(amount-10))
                uper{i}{j}(isnan(uper{i}{j})) = 0;
                vper{i}{j}(isnan(vper{i}{j})) = 0;
                dmoy.upermoy{i} = dmoy.upermoy{i} + (1/length(i:10:(amount-10))).*uper{i}{j};
                dmoy.vpermoy{i} = dmoy.vpermoy{i} + (1/length(i:10:(amount-10))).*vper{i}{j};
            end;
            dmoy.uperglob = dmoy.uperglob + (1/10).*dmoy.upermoy{i};
            dmoy.vperglob = dmoy.vperglob + (1/10).*dmoy.vpermoy{i};
        end
        %temp_varname = find(directory == '/');
        %varname = directory((temp_varname(end-1)+1):(temp_varname(end)-1));
        clearvars -except p s x y u v typevector xper yper uper vper typevectorper directory filenames u_filt v_filt typevector_filt dmoy typemask typemaskper varname dirlist ind
        cd(directory);
        name = ['ratio2_W32pix_' basename '_'];
        save([name date]);
        cd('..');
    end
end