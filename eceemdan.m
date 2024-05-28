% Function for ECEEMDAN,The current is a enhanced version of the CEEMDAN. Last version: 28 jun 2023
% Authors: Manuel Soto Calvo. manuel.sotocalvo@gmail.com; Han Soo Lee.leehs@hiroshima-u.ac.jp
% Paper: Enhanced complete ensemble EMD with superior noise handling capabilities: A robust signal decomposition method for power systems analysis DOI: 10.1002/eng2.12862
% Syntaxis
% [modes, its] = eceemdan(x,Nstd,NR,MaxIter,SNRFlag,window_size,filt_method,tolerance, useParpool)
%
% x: input signal
% Nstd: noise standard deviation amplitude
% NR: number of realizations
% MaxIter: maximum number of iterations
% SNRFlag: flag to adjust the noise amplitude
% window_size: size of the moving window for the moving standard deviation
% filt_method: filter method for smoothing the signal% Function for ECEEMDAN,The current is a enhanced version of the CEEMDAN. Last version: 28 jun 2023
% Authors: Manuel Soto Calvo. manuel.sotocalvo@gmail.com; Han Soo Lee.leehs@hiroshima-u.ac.jp
%
% Syntaxis
% [modes, its] = eceemdan(x,Nstd,NR,MaxIter,SNRFlag,window_size,filt_method,tolerance, useParpool)
%
% x: input signal
% Nstd: noise standard deviation
% NR: number of realizations
% MaxIter: maximum number of iterations
% SNRFlag: flag to adjust the noise amplitude
% window_size: size of the moving window for the moving standard deviation
% filt_method: filter method for smoothing the signal
% tolerance: threshold for stopping criteria
% useParpool: flag to decide if use parallel computation or not
% 'windows_size' must be a 2-elements vector which means the number of elements before and after of the central element.
% 'windows' must be > 0 && 'windows' < length (x - 1)
% 'tolerance' must be > 0, once the 'tolerance' criterion has been meet the calculaiton will stop 
%  
%************************************ Noise filtering methods ******************************************************************************  
%  'movmean' — Moving average over each window of A. This method is useful for reducing periodic trends in data.
% 
%  'movmedian' — Moving median over each window of A. This method is useful for reducing periodic trends in data when outliers are present.
% 
%  'gaussian' — Gaussian-weighted moving average over each window of A.
% 
%  'lowess' — Linear regression over each window of A. This method can be computationally expensive, but results in fewer discontinuities.
% 
%  'loess' — Quadratic regression over each window of A. This method is slightly more computationally expensive than 'lowess'.
% 
%  'rlowess' — Robust linear regression over each window of A. This method is a more computationally expensive version of the method 'lowess', but it is more robust to outliers.
% 
%  'rloess' — Robust quadratic regression over each window of A. This method is a more computationally expensive version of the method 'loess', but it is more robust to outliers.
% 
%  'sgolay' — Savitzky-Golay filter, which smooths according to a quadratic polynomial that is fitted over each window of A. This method can be more effective than other methods when the data varies rapidly.
%
%
% WARNING: for this code works it is necessary to include in the same
% directoy the file emd.m developed by Rilling and Flandrin.
% This file is available at http://perso.ens-lyon.fr/patrick.flandrin/emd.html


% OUTPUT
% modes: contain the obtained modes in a matrix with the rows being the modes        
% its: contain the sifting iterations needed for each mode for each realization (one row for each realization)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [modes, its]= eceemdan(x,Nstd,NR,MaxIter,SNRFlag,window_size,filt_method,tolerance, useParpool)

% Set default values
if ~exist('filt_method','var')
    filt_method = 'sgolay';
end
if ~exist('window_size','var')
    window_size = length(x)*0.001; 
end
if ~exist('tolerance','var')
    tolerance = 0;
end
if ~exist('SNRFlag','var')
    SNRFlag = 1;
end
if ~exist('MaxIter','var')
    MaxIter = 500;
end
if ~exist('NR','var')
    NR = 100;
end
if ~exist('Nstd','var')
    Nstd = 0.2;
end
if ~exist('useParpool','var')
    useParpool = 0;
end

tic
disp('Starting calculation')

x=x(:)';
desvio_x=std(x);
x=x/desvio_x;

% Calculating the moving standard deviation on x
std_signal = movstd(x, window_size);

modes=zeros(size(x));
temp=zeros(size(x));
aux=zeros(size(x));
iter=zeros(NR,round(log2(length(x))+5));

% Generating noise and calculating the first mode
disp('Generating noise and calculating the first mode')
if useParpool == 1
parfor i=1:NR
    white_noise{i} = Nstd .* std_signal .* randn(size(x));
    xi = x + white_noise{i};
    [mode_i, ~, it_i] = emd(xi,'MAXMODES',1,'MAXITERATIONS',MaxIter);
    mode_i = mode_i(1,:);
    aux = aux + (xi - mode_i) / NR;
    iter(i,1) = it_i;
end
else
for i = 1:NR
    white_noise{i} = Nstd .* std_signal .* randn(size(x));
    xi = x + white_noise{i};
    [mode_i, ~, it_i] = emd(xi,'MAXMODES',1,'MAXITERATIONS',MaxIter);
    mode_i = mode_i(1,:);
    aux = aux + (xi - mode_i) / NR;
    iter(i,1) = it_i;
    
end
end

modes= x-aux; % saves the first mode
medias = aux;
k=1;
aux=zeros(size(x));
es_imf = min(size(emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter)));

last_modes = []; % Empty matriz to storage previous modes

% Main loop
while es_imf > 1
    aux=zeros(size(x));
    for i=1:NR
        tam = size(white_noise{i});
        if tam(1) >= k+1
            noise_i = white_noise{i}(k+1,:);

            if SNRFlag == 1
                noise_i = noise_i / std(noise_i); % Adjusting the std of the noise
            end
            noise_i = Nstd * noise_i;
            signal_i = medias(end,:) + std(medias(end,:)) * noise_i;

            try
                [mode_i,~,it_i]=emd(signal_i,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            catch
                disp('emd function error, setting it_i to 0')
                it_i=0;
                mode_i = emd(signal_i,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            end

            % Filtering the noise
            signal_i = smoothdata(signal_i,filt_method,window_size);

            [mode_i, ~, it_i] = emd(signal_i,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            mode_i = mode_i(end,:);

            % Update noise estimation
            noise_estimation_i = std(signal_i - mode_i);
            white_noise{i} = Nstd .* noise_estimation_i .* randn(size(x));
        else
            signal_i = medias(end,:);

            % Filtering the noise
            signal_i = smoothdata(signal_i,filt_method,window_size);

            try
                [mode_i, ~, it_i] = emd(signal_i,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            catch
                disp('emd function error, setting it_i to 0')
                it_i=0;
                mode_i = emd(signal_i,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            end
            mode_i = mode_i(end,:);
        end
        aux = aux + mode_i / NR;
        iter(i, k+1) = it_i;
    end

    new_mode = medias(end,:) - aux;

    modes = [modes; new_mode];

    medias = [medias; aux];

    % Adding a stopping criterion based on the tolerance
    if ~isempty(last_modes) && max(abs(new_mode - last_modes)) < tolerance
        break;
    end

    % Updating modes
    last_modes = new_mode;
    k = k + 1;
    es_imf = min(size(emd(medias(end,:), 'MAXMODES',1, 'MAXITERATIONS',MaxIter)));
end

modes = [modes; medias(end,:)];

modes = modes * desvio_x;
its = iter;

disp('Decomposition completed')

toc

% Showing result
figure
subplot(size(modes,1),1,1)
for j = 1:size(modes,1)
    subplot(size(modes,1),1,j)
    plot(modes(j,:))
    hold on
    if j < size(modes,1)
        ylabel(["IMF ", num2str(j)])
    else
        ylabel("Residual")

    end
    xlim([1 length(modes)])
end

end



