function process_dir(Dirbase, reprocess)

    %Dirbase = '/Volumes/labshared3-1/takumi/2018_02_01';
    Dir = [Dirbase '/Tiff_folder'];
    cd(Dir)

    Dirs = dir('PIV*File');
    dirlist = {Dirs.name};

    disp(dirlist)


    % Use below to process in multiple ways
    Dts = [1]; % process PIV from image n and image n+Dt...
    % steps = [10, 20, 30, 40, 50, 60, 70, 80, 90];
    steps = [2]; %process image n & imange n+Dt pair, then process n+step & n+Dt+step...
    Ws = [16];
    max = 0;  % Process images till max. If 0, process all
    % max = 59;

    % for j=1:length(Dts)
    %     for i=1:length(Ws)
    %         treatpiv_t(Dirbase,dirlist,Dts(j),Ws(i),steps(i), max);
    %     end
    % end

    max = 0;
    for j=1:length(Dts)
        for i=1:length(Ws)
            treatpiv_t(Dirbase,dirlist,Dts(j),Ws(i),steps(i), max, reprocess);
        end
    end



