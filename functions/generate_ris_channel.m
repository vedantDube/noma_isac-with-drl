function [H_eff, beta_s_eff, Phi] = generate_ris_channel(para, H_direct, beta_s_direct, r_users, th_users)
%Generate RIS-assisted effective communication and sensing channels
%   [H_eff, beta_s_eff, Phi] = generate_ris_channel(para, H_direct, beta_s_direct, r_users, th_users)
%
% Models a passive RIS assisting the near-field ISAC system.
% The RIS applies a co-phased (maximal-ratio) phase-shift matrix Phi to
% coherently combine the BS->RIS->user/target paths with direct paths.
%
% Inputs:
%   para         - system parameters (must include para.M, para.r_ris,
%                  para.theta_ris, para.r_s, para.theta_s)
%   H_direct     - direct BS->user channel [N x K]
%   beta_s_direct- direct sensing channel gain (complex scalar)
%   r_users      - user distances [K x 1] (optional for legacy support)
%   th_users     - user angles [K x 1] (optional for legacy support)
%
% Outputs:
%   H_eff        - effective channel [N x K] (direct + RIS-cascaded)
%   beta_s_eff   - effective sensing gain (direct + RIS-cascaded, scalar)
%   Phi          - RIS phase-shift matrix [M x M] (diagonal)

    N = para.N;
    M = para.M;
    K = para.K;
    lambda = para.c / para.f;
    
    % FIX: RIS element spacing should be constant lambda/2, not shrink with M
    d_ris  = 0.5 * lambda;   % half-wavelength RIS element spacing

    %% ---- RIS Steering Vectors ----
    % BS antenna positions (same as beamfocusing.m convention)
    n_bs = (-(N-1)/2 : (N-1)/2)' * para.d;   % [N x 1]

    % RIS element positions (ULA centred at origin)
    m_ris = (-(M-1)/2 : (M-1)/2)' * d_ris;   % [M x 1]

    % Helper: near-field array response for a generic ULA
    %   pos  - element positions [L x 1]
    %   r0   - distance of source
    %   th   - angle of source (rad)
    nf_steer = @(pos, r0, th) exp(-1i*2*pi/lambda * ...
        (sqrt(r0^2 + pos.^2 - 2*r0*pos*cos(th)) - r0));

    %% ---- BS -> RIS Channel (H_br) [M x N] ----
    a_bs_ris  = nf_steer(n_bs,  para.r_ris, para.theta_ris);   % [N x 1]
    a_ris_bs  = nf_steer(m_ris, para.r_ris, para.theta_ris);   % [M x 1]

    % Path loss for BS-RIS link
    beta_br = sqrt(para.rho_0) / para.r_ris * ...
              exp(-1i*2*pi*para.f/para.c * para.r_ris);
    H_br = beta_br * (a_ris_bs * a_bs_ris.');   % [M x N]

    %% ---- RIS -> User Channels (H_ru) [K x M] ----
    % Use provided user locations if available, otherwise fallback to consistent randomization
    if nargin < 4 || isempty(r_users) || isempty(th_users)
        % Backward compatibility fallback: deterministic randomization seeded from K
        % (Note: we use a fixed seed to ensure consistency if not provided)
        prev_rng = rng(2024 + K, 'twister'); 
        Rayleigh_distance = 2*para.D^2/lambda;
        r_users   = rand(K,1) * Rayleigh_distance;
        th_users  = rand(K,1) * pi;
        rng(prev_rng);
    end

    H_ru = zeros(K, M);   % [K x M]
    for k = 1:K
        beta_ru = sqrt(para.rho_0) / r_users(k) * ...
                  exp(-1i*2*pi*para.f/para.c * r_users(k));
        a_ris_uk = nf_steer(m_ris, r_users(k), th_users(k));   % [M x 1]
        H_ru(k,:) = beta_ru * a_ris_uk.';                        % [1 x M]
    end

    %% ---- RIS Phase Shift: Co-phased for sensing target ----
    % Align RIS toward sensing target: maximize |a_ris_target' * Phi * a_ris_bs|
    a_ris_target = nf_steer(m_ris, para.r_s, para.theta_s);   % [M x 1]
    % Optimal co-phased: phi_m = -angle(a_ris_target(m)) + angle(a_ris_bs(m))
    phi_vec = -angle(a_ris_target) + angle(a_ris_bs);
    Phi = diag(exp(1i * phi_vec));   % [M x M]

    %% ---- Cascaded BS -> RIS -> User Channel ----
    % G_casc [K x N] = H_ru * Phi * H_br
    G_casc = H_ru * Phi * H_br;   % [K x N]

    % Effective user channel [N x K]: direct + cascaded
    H_eff = H_direct + G_casc.';   % [N x K]

    %% ---- Effective Sensing Gain ----
    % Direct sensing gain
    beta_direct = beta_s_direct;

    % RIS-assisted sensing path:
    %   BS -> RIS: a_ris_bs, path-loss beta_br
    %   RIS -> target (forward): a_ris_target, path-loss beta_rt
    %   target -> RIS (back-scatter): same path (monostatic)
    %   RIS -> BS: same as BS -> RIS (reciprocal)
    beta_rt = sqrt(para.rho_0) / para.r_s * ...
              exp(-1i*2*pi*para.f/para.c * para.r_s);

    % RIS gain factor (scalar): a_ris_target' * Phi * a_ris_bs
    ris_gain = a_ris_target' * Phi * a_ris_bs;    % scalar

    % Combined RIS-path sensing coefficient
    beta_ris = beta_br * beta_rt * ris_gain;      % scalar

    % Effective sensing gain (direct + RIS path)
    beta_s_eff = beta_direct + beta_ris;
end

