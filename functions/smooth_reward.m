function y = smooth_reward(x, window)
%SMOOTH_REWARD  Causal moving-average smoothing for reward curves
%   y = smooth_reward(x, window)
%
% Inputs:
%   x      - reward vector [T x 1] or [1 x T]
%   window - smoothing window length (integer >= 1)
%
% Output:
%   y - smoothed reward, same size as x

    x = x(:);   % ensure column vector
    T = length(x);
    y = zeros(T, 1);
    for t = 1:T
        idx_start = max(1, t - window + 1);
        y(t) = mean(x(idx_start:t));
    end

    % Restore original orientation
    if size(x, 2) > size(x, 1)
        y = y.';
    end
end
