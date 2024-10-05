% Function for ECEEMDAN,The current is a enhanced version of the CEEMDAN. Last version: 5 oct 2024
% Authors: Manuel Soto Calvo. manuel.sotocalvo@gmail.com; Han Soo Lee.leehs@hiroshima-u.ac.jp
% Paper: Enhanced complete ensemble EMD with superior noise handling capabilities: A robust signal decomposition method for power systems analysis DOI: 10.1002/eng2.12862
% Syntaxis
% [modes, its] = eceemdan(x, NR, MaxIter, tolerance, useParpool, window_size, Nstd)
%
% x: input signal

% NR: number of realizations
% MaxIter: maximum number of iterations
% tolerance: threshold for stopping criteria
% useParpool: flag to decide if use parallel computation or not
% window_size: size of the moving window for the moving standard deviation.It is computed heuristically if not inputed
% Nstd: noise standard deviation amplitude. It is computed heuristically if not inputed

% WARNING: for this code works it is necessary to include in the same
% directoy the file emd.m developed by Rilling and Flandrin.
% This file is available at http://perso.ens-lyon.fr/patrick.flandrin/emd.html


% OUTPUT
% modes: contain the obtained modes in a matrix with the rows being the modes        
% its: contain the sifting iterations needed for each mode for each realization (one row for each realization)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [modes, its] = eceemdan(x, NR, MaxIter, tolerance, useParpool, window_size, Nstd)
    % Set default values
    if ~exist('tolerance', 'var')
        tolerance = 0;
    end
    if ~exist('MaxIter', 'var')
        MaxIter = 500;
    end
    if ~exist('useParpool', 'var')
        useParpool = 0;
    end

    tic
    disp('Starting calculation')

    x = x(:)';
    desvio_x = std(x);
    x = x / desvio_x;

    if ~exist('window_size', 'var') || isempty(window_size)   
        window_size = calculate_window_size(x);
    end
    disp(['Window size set to: ', num2str(window_size)])
   
    
    if ~exist('Nstd', 'var') || isempty(Nstd)
        Nstd = calculate_noise_intensity(x, window_size);
    end
    disp(['Noise amplitude set to: ', num2str(Nstd)])

    std_signal = movstd(x, window_size);
    modes = zeros(size(x));
    aux = zeros(size(x));
    iter = zeros(NR, round(log2(length(x)) + 5));

    disp('Generating noise and calculating the first mode')
    if useParpool == 1
        parfor i = 1:NR
            [white_noise{i}, xi] = generate_noise(x, Nstd, std_signal);
            [mode_i, it_i] = emd_mode(xi, MaxIter);
            aux = aux + (xi - mode_i) / NR;
            iter(i, 1) = it_i;
        end
    else
        for i = 1:NR
            [white_noise{i}, xi] = generate_noise(x, Nstd, std_signal);
            [mode_i, it_i] = emd_mode(xi, MaxIter);
            aux = aux + (xi - mode_i) / NR;
            iter(i, 1) = it_i;
        end
    end

    modes = x - aux; % saves the first mode
    medias = aux;
    k = 1;
    aux = zeros(size(x));
    es_imf = min(size(emd(medias(end, :), 'MAXMODES', 1, 'MAXITERATIONS', MaxIter)));

    last_modes = []; % Empty matrix to store previous modes

    while es_imf > 1
        aux = zeros(size(x));
        for i = 1:NR
            tam = size(white_noise{i});
            if tam(1) >= k + 1
                noise_i = white_noise{i}(k + 1, :);
                noise_i = Nstd * noise_i;
                signal_i = medias(end, :) + std(medias(end, :)) * noise_i;
                window_size = calculate_window_size(signal_i);                 

                [mode_i, it_i] = emd_with_retry(signal_i, MaxIter, window_size);                
                aux = aux + mode_i / NR;
                iter(i, k + 1) = it_i;
            else
                signal_i = medias(end, :);
                window_size = calculate_window_size(signal_i);                
                [mode_i, it_i] = emd_with_retry(signal_i, MaxIter, window_size);               
                aux = aux + mode_i / NR;
                iter(i, k + 1) = it_i;
            end
           
        end

        new_mode = medias(end, :) - aux;

        % Agrupación de modos
        if k > 1
            new_mode = group_modes(new_mode, modes);
        end

        modes = [modes; new_mode];
        medias = [medias; aux];

        if stopping_criterion(new_mode, last_modes, tolerance)
            break;
        end

        last_modes = new_mode;
        k = k + 1;
        es_imf = min(size(emd(medias(end, :), 'MAXMODES', 1, 'MAXITERATIONS', MaxIter)));
    end

    modes = [modes; medias(end, :)];
    modes = modes * desvio_x;
    its = iter;    

    disp('Decomposition completed')

    toc

    plot_results(modes);

end

function [white_noise, xi] = generate_noise(x, Nstd, std_signal)
    white_noise = Nstd .* std_signal .* randn(size(x));
    xi = x + white_noise;
end

function [mode_i, it_i] = emd_mode(xi, MaxIter)
    [mode_i, ~, it_i] = emd(xi, 'MAXMODES', 1, 'MAXITERATIONS', MaxIter);
    mode_i = mode_i(1, :);
end

function window_size = calculate_window_size(x)
    % Calculate signal properties using simplified_autocorr
    [acf, ~, first_zero_crossing] = simplified_autocorr(x);

    % Estimate dominant period
    [pxx, f] = periodogram(x);
    [~, max_pxx_idx] = max(pxx(2:end));  
    max_pxx_idx = max_pxx_idx + 1;  
    if f(max_pxx_idx) > 0
        dominant_period = 1 / f(max_pxx_idx);
    else
        dominant_period = length(x);  
    end
    
    % Estimate noise level
    noise_level = estimate_noise_level(x);
    
    % Calculate window size based on signal properties
    window_size = round(log(dominant_period) * (1 + noise_level) * first_zero_crossing);
end

function [acf, lags, first_zero_crossing] = simplified_autocorr(y, numLags)
    % Ensure y is a column vector
    y = y(:);
    
    % Remove mean
    y = y - mean(y, 'omitnan');
    
    % Get effective sample size
    N = sum(~isnan(y));
    
    % Set default numLags if not provided
    if nargin < 2 || isempty(numLags)
        numLags = min(20, N-1);
    end
    
    % Compute ACF
    if N < length(y)  % Missing data
        acf = nan(numLags+1, 1);
        for j = 0:numLags
            cross = y(1:end-j) .* y(j+1:end);
            iNonNaN = ~isnan(cross);
            if any(iNonNaN)
                T = sum(iNonNaN) + sum(~isnan(y(1:j)));
                acf(j+1) = sum(cross, 'omitnan') / T;
            end
        end
    else  % No missing data
        nFFT = 2^(nextpow2(length(y))+1);
        F = fft(y, nFFT);
        F = F .* conj(F);
        acf = ifft(F);
        acf = acf(1:(numLags+1));  % Retain nonnegative lags
        acf = real(acf);
    end
    
    % Normalize
    acf = acf ./ acf(1);
    
    % Generate lags
    lags = (0:numLags)';
    
    % Calculate first zero crossing
    zero_crossings = find(diff(sign(acf)) ~= 0);
    if ~isempty(zero_crossings)
        first_zero_crossing = zero_crossings(1) - 1;  % Adjust for 0-based lag
    else
        first_zero_crossing = length(acf) - 1;  % If no crossing, use max lag
    end
end


function noise_level = estimate_noise_level(x)
    % Estimate noise level using median absolute deviation
    diff_x = diff(x);
    mad_x = median(abs(diff_x - median(diff_x)));
    noise_level = mad_x / 0.6745;  % Assuming Gaussian noise
    noise_level = noise_level / std(x);  % Normalize by signal std
end

function Nstd = calculate_noise_intensity(x, window_size)
    % Calculate moving standard deviation
    std_signal = movstd(x, window_size);
    
    % Calculate spectral entropy
    [pxx, f] = pwelch(x);
    pxx_norm = pxx / sum(pxx);
    spectral_entropy = -sum(pxx_norm .* log(pxx_norm + eps));
    
    % Calculate signal-to-noise ratio estimate
    snr_estimate = 10 * log10(var(x) / var(diff(x)));
    
    % Calculate scale factor based on spectral entropy and SNR
    scale_factor = (1 - spectral_entropy / log2(length(pxx))) * (1 + exp(-snr_estimate/10));
    
    % Calculate base noise intensity
    base_intensity = 1 - mean(std_signal);
    
    % Apply adaptive scaling
    Nstd = base_intensity * scale_factor;
    
    % Ensure Nstd is within a reasonable range
    Nstd = max(0.01, min(Nstd, 0.9));
    
    disp(['Calculated noise intensity: ', num2str(Nstd)]);
end

function [mode_i, it_i] = emd_with_retry(signal_i, MaxIter, window_size)
    max_retries = 3;
    retry_count = 0;
    
    while retry_count < max_retries
        try
            if window_size > 1
                signal_i = smoothdata(signal_i, 'sgolay', window_size);
            else
                signal_i = smoothdata(signal_i, 'sgolay');
            end
            
            [mode_i, ~, it_i] = emd(signal_i, 'MAXMODES', 1, 'MAXITERATIONS', MaxIter);
            mode_i = mode_i(end, :);
            return;
        catch ME
            retry_count = retry_count + 1;
            warning('EMD failed. Retry %d of %d. Error: %s', retry_count, max_retries, ME.message);
            
            % Reducir el tamaño de la ventana en caso de error
            window_size = max(1, round(window_size * 0.8));
            
            % Aumentar MaxIter en caso de no convergencia
            if contains(ME.message, 'convergence')
                MaxIter = round(MaxIter * 1.5);
            end
        end
    end
    
    % Si todos los intentos fallan, devolver una aproximación
    warning('All EMD attempts failed. Returning approximation.');
    mode_i = signal_i - mean(signal_i);
    it_i = 0;
end
function stop = stopping_criterion(new_mode, last_modes, tolerance)
    stop = ~isempty(last_modes) && max(abs(new_mode - last_modes)) < tolerance;
end

function plot_results(modes)
    figure
    subplot(size(modes, 1), 1, 1)
    for j = 1:size(modes, 1)
        subplot(size(modes, 1), 1, j)
        plot(modes(j, :))
        hold on
        if j < size(modes, 1)
            ylabel(["IMF ", num2str(j)])
        else
            ylabel("Residual")
        end
        xlim([1 length(modes)])
    end
end

function new_mode = group_modes(new_mode, modes)
    threshold_corr = 0.8;  
    threshold_energy = 0.2;  

    for i = 1:size(modes, 1)
        % Calcular la correlación cruzada normalizada
        corr_coef = max(xcorr(new_mode, modes(i,:), 'coeff'));
        
        % Calcular la diferencia de energía normalizada
        energy_diff = abs(sum(new_mode.^2) - sum(modes(i,:).^2)) / sum(new_mode.^2);
        
        if corr_coef > threshold_corr && energy_diff < threshold_energy
            % Combinar los modos usando una media ponderada
            weight = corr_coef * (1 - energy_diff);
            new_mode = (weight * modes(i,:) + new_mode) / (weight + 1);
        end
    end
end


