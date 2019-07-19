function process_dir(Dirbase, reprocess)

    %Dirbase = '/Volumes/labshared3-1/takumi/2018_02_01';
    Dir = [Dirbase '/Tiff_folder'];
%     Dir = [Dirbase '/'];
    cd(Dir)

    Dirs = dir('*800_step1');
%     Dirs = dir('npt50000_lt20p0_pbc*test');
    dirlist = {Dirs.name};

    disp(dirlist)


    % Use below to process in multiple ways
    Dts = [1]; % process PIV from image n and image n+Dt...
    % steps = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    steps = [2]; %process image n & imange n+Dt pair, then process n+step & n+Dt+step...
    Ws = [16];
    maxind = 0;  % Process images till maxind. If 0, process all
    bitnumber = 16; % quality of image
    % max = 59;

%     for j=1:length(Dts)
%         for i=1:length(Ws)
%             treatpiv_t(Dirbase,dirlist,Dts(j),Ws(i),steps(i), max);
%         end
%     end

%     max = 400;
    for j=1:length(Dts)
        for i=1:length(Ws)
            treatpiv_t(Dirbase,dirlist,Dts(j),Ws(i),steps(i), maxind, bitnumber, reprocess);
        end
    end



