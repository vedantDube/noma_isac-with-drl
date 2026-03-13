%% ris_sweep.m
% Sweeps RIS element count M = [50,100,200,300,400].
% For each M: generates RIS-assisted channels, trains TD3 / PPO / DDPG,
% smooths reward histories, saves agent structs + results to
%   results/ris_M<N>/results.mat
%
% Usage:  Run this script directly from the project root in MATLAB.
% After completion, run plot_ris_results.m to generate all plots.

clc; close all;
addpath('./functions');

%% ---- Shared channel (fixed across all RIS experiments) ----
rng(42);   % reproducible channel
para_base = para_init();
[H_direct, G, beta_s_direct, r_users, theta_users, r_s, theta_s] = ...
    generate_channel(para_base);

% Sort users by channel gain (ascending) for NOMA
cg = arrayfun(@(k) norm(H_direct(:,k))^2, 1:para_base.K);
[~, idx] = sort(cg, 'ascend');
H_direct = H_direct(:, idx);

%% ---- Training options ----
drl_options.num_episodes  = 1000;  % increased from 2000 for deeper learning
drl_options.max_episodes  = 1000;
drl_options.hidden_dim    = 256;
drl_options.warmup_steps  = 1500;
drl_options.channel_noise = 0.005;
drl_options.explore_noise = 0.3;  % increased from 0.2 for more exploration
drl_options.batch_size    = 128;
drl_options.lr_actor      = 1e-3;    % learning rate for actor network
drl_options.lr_critic     = 1e-3;    % learning rate for critic network
drl_options.buffer_size   = 100000;
smooth_window             = 50;    % moving-average window for reward smoothing

%% ---- RIS sweep ----
M_values = [50, 100, 200, 300, 400];
num_reps = 5;                    % number of independent trials per M

n_M = length(M_values);

% allocate storage for statistics
CRBs_td3  = nan(num_reps, n_M);
CRBs_ppo  = nan(num_reps, n_M);
CRBs_ddpg = nan(num_reps, n_M);

SumRates_td3  = nan(num_reps, n_M);
SumRates_ppo  = nan(num_reps, n_M);
SumRates_ddpg = nan(num_reps, n_M);

for rep = 1:num_reps
    fprintf('\n######## Trial %d / %d ########\n', rep, num_reps);
    for m_idx = 1:n_M
        M = M_values(m_idx);
        fprintf('\n============================================================\n');
        fprintf('  RIS Sweep  |  M = %d elements  (trial %d)\n', M, rep);
        fprintf('============================================================\n');

        %% common seed per (rep,M) so experiments are repeatable
        seed = 1000*rep + M;
        rng(seed);

        %% Set RIS size in para
        para = para_base;
        para.M = M;

        %% Generate RIS-assisted effective channels
        [H_eff, beta_s_eff, Phi] = generate_ris_channel(para, H_direct, beta_s_direct, r_users, theta_users);

        %% --- TD3 ---
        fprintf('\n--- TD3 (M=%d, rep=%d) ---\n', M, rep);
        [Rx_td3, f_td3, CRB_td3, reward_td3_raw, agent_td3] = ...
            TD3_optimize(para, H_eff, beta_s_eff, drl_options, seed);
        reward_td3 = smooth_reward(reward_td3_raw, smooth_window);

        %% --- PPO ---
        fprintf('\n--- PPO (M=%d, rep=%d) ---\n', M, rep);
        [Rx_ppo, f_ppo, CRB_ppo, reward_ppo_raw, agent_ppo] = ...
            PPO_optimize(para, H_eff, beta_s_eff, drl_options, seed);
        reward_ppo = smooth_reward(reward_ppo_raw, smooth_window);

        %% --- DDPG ---
        fprintf('\n--- DDPG (M=%d, rep=%d) ---\n', M, rep);
        [Rx_ddpg, f_ddpg, CRB_ddpg, reward_ddpg_raw, agent_ddpg] = ...
            DDPG_optimize(para, H_eff, beta_s_eff, drl_options, seed);
        reward_ddpg = smooth_reward(reward_ddpg_raw, smooth_window);

        %% --- compute sum rates for this trial ---
        rate_td3  = rate_calculator(para, H_eff, Rx_td3,  f_td3);
        rate_ppo  = rate_calculator(para, H_eff, Rx_ppo,  f_ppo);
        rate_ddpg = rate_calculator(para, H_eff, Rx_ddpg, f_ddpg);

        CRBs_td3(rep,m_idx)  = CRB_td3;
        CRBs_ppo(rep,m_idx)  = CRB_ppo;
        CRBs_ddpg(rep,m_idx) = CRB_ddpg;

        SumRates_td3(rep,m_idx)  = sum(rate_td3);
        SumRates_ppo(rep,m_idx)  = sum(rate_ppo);
        SumRates_ddpg(rep,m_idx) = sum(rate_ddpg);

        %% --- Save per-run results (for diagnostics) ---
        save_dir = sprintf('./results/ris_M%d', M);
        if ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        save_path = fullfile(save_dir, sprintf('results_rep%d.mat', rep));
        save(save_path, ...
            'M', 'para', 'Phi', 'H_eff', 'beta_s_eff', ...
            'reward_td3',  'reward_td3_raw',  'CRB_td3',  'agent_td3',  'Rx_td3',  'f_td3',  ...
            'reward_ppo',  'reward_ppo_raw',  'CRB_ppo',  'agent_ppo',  'Rx_ppo',  'f_ppo',  ...
            'reward_ddpg', 'reward_ddpg_raw', 'CRB_ddpg', 'agent_ddpg', 'Rx_ddpg', 'f_ddpg');
        fprintf('\nSaved results to %s\n', save_path);
        fprintf('CRB  ->  TD3: %.4f  |  PPO: %.4f  |  DDPG: %.4f\n', ...
            CRB_td3, CRB_ppo, CRB_ddpg);
    end
end

% aggregate statistics
CRB_mean_td3  = mean(CRBs_td3,1);
CRB_std_td3   = std(CRBs_td3,0,1);
CRB_mean_ppo  = mean(CRBs_ppo,1);
CRB_std_ppo   = std(CRBs_ppo,0,1);
CRB_mean_ddpg = mean(CRBs_ddpg,1);
CRB_std_ddpg  = std(CRBs_ddpg,0,1);

% save aggregated results for plotting
agg_path = './results/ris_aggregate.mat';
save(agg_path, 'M_values', ...
    'CRB_mean_td3','CRB_std_td3', ...
    'CRB_mean_ppo','CRB_std_ppo', ...
    'CRB_mean_ddpg','CRB_std_ddpg', ...
    'SumRates_td3','SumRates_ppo','SumRates_ddpg');

fprintf('\n\n===  RIS Sweep Complete! (aggregated saved to %s)  ===\n', agg_path);
fprintf('Run plot_ris_results.m to generate comparison plots.\n');

fprintf('\n\n===  RIS Sweep Complete!  ===\n');
fprintf('Run plot_ris_results.m to generate comparison plots.\n');
