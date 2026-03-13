function [para] = para_init()
%Construct a struct of the initial values for all the parameters 
%  [values] = para_init()
%Inputs:
%   None
%Outputs:
%   para: a struct
%Date: 14/61/2023
%Author: Zhaolin Wang


para.N = 65; % the number of transmit antennas
para.N_RF = 5; % the number of RF chains
para.T = 128; % the length of coherent time block

para.Pt = 10^(20/10); % overall transmit power (dBm)
para.K = 4; % user number
para.noise = 10^(-60/10); % noise power in dBm
para.Rmin = 0.5; % minimum communication rate (bps/Hz) (reduced from 1.0 for easier training)

para.c = 3e8; % speed of light in free space
para.f = 28e9; % carrier frequency
para.D = 0.5; % antenna aperture
para.d = para.D/(para.N-1); % antenna spacing

para.rho_0 = 1/(4*pi*para.f/para.c); % reference pathloss

% location of target
para.r_s = 20;
para.theta_s = 45*pi/180;

% RIS parameters (M overridden in ris_sweep.m per experiment)
para.M      = 50;                        % number of RIS elements (default)
para.r_ris  = 15;                        % RIS distance from BS (m)
para.theta_ris = 30*pi/180;              % RIS angle from BS (rad)
% RIS element spacing computed dynamically in generate_ris_channel.m

end

