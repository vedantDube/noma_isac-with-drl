function [rate] = rate_calculator(para, H, Rx, f)
%MUSIC algorithm
%   [rate] = rate_calculator(para, H, Rx, f)
%Inputs:
%   para: structure of the initial parameters
%   H: communication channels
%   Rx: covariance matrix of transmit signal
%   f: beamformers of communication signals
%Outputs:
%   rate: communication rates for all users
%Date: 16/04/2025
%Author: Zhaolin Wang

rate = zeros(para.K, 1);
for k = 1:para.K
    hk = H(:,k);
    fk = f(:,k);
    % NOMA SIC: user k cancels weaker users 1..k-1
    % Only interference from stronger users k+1..K
    % Sensing signal is known/deterministic -> cancelable at receivers
    interference = 0;
    for i = k+1:para.K
        interference = interference + abs(hk.' * f(:,i))^2;
    end
    SINR_k = abs(hk.'*fk)^2 / (interference + 1);
    rate(k) = log2(1 + SINR_k);
end


end

