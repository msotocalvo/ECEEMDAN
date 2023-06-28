% Function for CEEMDAN

% WARNING: for this code works it is necessary to include in the same
% directoy the file emd.m developed by Rilling and Flandrin.
% This file is available at %http://perso.ens-lyon.fr/patrick.flandrin/emd.html
% We use the default stopping criterion.
% We use the last modification: 3.2007

%   Syntax

% modes=ceemdan(x,Nstd,NR,MaxIter,SNRFlag)
% [modes its]=ceemdan(x,Nstd,NR,MaxIter,SNRFlag)

%   Description

% OUTPUT
% modes: contain the obtained modes in a matrix with the rows being the modes        
% its: contain the sifting iterations needed for each mode for each realization (one row for each realization)

% INPUT
% x: signal to decompose
% Nstd: noise standard deviation
% NR: number of realizations
% MaxIter: maximum number of sifting iterations allowed.
% SNRFlag: if equals 1, then the SNR increases for every stage, as in [1].
%          if equals 2, then the SNR is the same for all stages, as in [2]. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The current is an improved version, introduced in:

%[1] Colominas MA, Schlotthauer G, Torres ME. "Improve complete ensemble EMD: A suitable tool for biomedical signal processing" 
%       Biomedical Signal Processing and Control vol. 14 pp. 19-29 (2014)

%The original CEEMDAN algorithm was first introduced at ICASSP 2011, Prague, Czech Republic

%The authors will be thankful if the users of this code reference the work
%where the algorithm was first presented:

%[2] Torres ME, Colominas MA, Schlotthauer G, Flandrin P. "A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise"
%       Proc. 36th Int. Conf. on Acoustics, Speech and Signa Processing ICASSP 2011 (May 22-27, Prague, Czech Republic)

%Author: Marcelo A. Colominas
%contact: macolominas@bioingenieria.edu.ar
%Last version: 25 feb 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The current is a enhanced version
%*************************************************************************************************************************************************************************************************************
% 'windows_size' must be a 2-elements vector which means the number of elements before and after of the central element.
% 'windows' must be > 0 && 'windows' < length (x - 1)
% 'tolerance' must be > 0, once the 'tolerance' criterion has been meet the calculaiton will stop 
%  Noise filtering methods:  
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
%**************************************************************************************************************************************************************************************************************

function [modes,its]= eceemdan(x,Nstd,NR,MaxIter,SNRFlag,window_size,filt_method,tolerance)
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

delete(gcp('nocreate'))
parpool(); % Opening a parpool

% Generating noise and calculating the first mode 
disp('Generating noise and calculating the first mode')
parfor i=1:NR
    white_noise{i} = Nstd .* std_signal .* randn(size(x));
    xi = x + white_noise{i};
    [temp, ~, it] = emd(xi,'MAXMODES',1,'MAXITERATIONS',MaxIter);
    temp = temp(1,:);
    aux = aux + (xi - temp) / NR;
    iter(i,1) = it;
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
            noise = white_noise{i}(k+1,:);
            
            if SNRFlag == 2
                noise = noise / std(noise); % Adjusting the std of the noise
            end
            noise = Nstd * noise;
            signal = medias(end,:) + std(medias(end,:)) * noise;

            try
                [temp,~,it]=emd(medias(end,:)+std(medias(end,:)) * noise,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            catch    
                it=0;
                temp = emd(medias(end,:) + std(medias(end,:)) * noise,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            end;

            % Filtering the noise
            signal = smoothdata(signal,filt_method,window_size);
 
            [temp, ~, it] = emd(signal(end,:) + std(medias(end,:)) * noise,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            temp = temp(end,:);

            % Update noise estimation
            noise_estimation = std(signal - temp);
            white_noise{i} = Nstd .* noise_estimation .* randn(size(x));
        else
            signal = medias(end,:);

            % Filtering the noise
            signal = smoothdata(signal,filt_method,window_size);

            try
                [temp, ~, it] = emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter);
                
            catch
                temp = emd(medias(end,:),'MAXMODES',1,'MAXITERATIONS',MaxIter);
                it=0;

            end
            temp = temp(end,:);

        end
        aux = aux + temp / NR;
        iter(i, k+1) = it;
        
    end
   
    new_mode = medias(end,:) - aux;

         % Validation of the new_mode based on the energy relevancy 
        new_mode_energy = max(new_mode);
        if new_mode_energy >= std(cell2mat(white_noise))
            % If the energy of the new_mode is greter than treshold is included
            modes = [modes; new_mode];
                        
        else
            % Otherway is fusioned with the previous mode
            modes(end,:) = modes(end,:) + new_mode;
            
        end

%     modes = [modes; new_mode];
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
delete(gcp('nocreate'))
toc
