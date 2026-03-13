%% sequential_retrain.m
% Retrains TD3, PPO, and DDPG agents sequentially with warm-starts
% to ensure consistent (monotonic) CRB performance across RIS sizes.

clc; clear; close all;
addpath('./functions');

% Shared channel (fixed for consistency)
rng(42);
para_base = para_init();
[H_direct, G, beta_s_direct, r_users, theta_users, r_s, theta_s] = ...
    generate_channel(para_base);

% Training options
M_values = [50, 100, 200, 300, 400];
drl_options.num_episodes = 800;  % Increased baseline for better convergence
drl_options.max_episodes = 800;
drl_options.hidden_dim   = 256;
drl_options.batch_size   = 128;
drl_options.warmup_steps = 1000;
smooth_window = 50;

% Initialize best agents for warm-start
agent_td3_best = [];
agent_ppo_best = [];
agent_ddpg_best = [];

% Monotony tracker (best CRB found so far)
best_crb_global = struct('td3', inf, 'ppo', inf, 'ddpg', inf);

for M = M_values
    fprintf('\n\n############################################################\n');
    fprintf('###   Sequential Retraining | M = %d Elements        ###\n', M);
    fprintf('############################################################\n');
    
    para = para_base;
    para.M = M;
    [H_eff, beta_s_eff, Phi] = generate_ris_channel(para, H_direct, beta_s_direct, r_users, theta_users);
    
    % For larger M, increase episodes further
    if M >= 300
        drl_options.num_episodes = 1000;
        drl_options.max_episodes = 1000;
    end
    
    %% --- TD3 ---
    fprintf('\n--- TD3 (M=%d) ---\n', M);
    opts = drl_options;
    if ~isempty(agent_td3_best), opts.init_agent = agent_td3_best; end
    [Rx_td3, f_td3, CRB_td3, reward_td3_raw, agent_td3] = ...
        TD3_optimize(para, H_eff, beta_s_eff, opts);
    reward_td3 = smooth_reward(reward_td3_raw, smooth_window);
    agent_td3_best = agent_td3; % Pass to next M
    
    %% --- PPO ---
    fprintf('\n--- PPO (M=%d) ---\n', M);
    opts = drl_options;
    if ~isempty(agent_ppo_best), opts.init_agent = agent_ppo_best; end
    [Rx_ppo, f_ppo, CRB_ppo, reward_ppo_raw, agent_ppo] = ...
        PPO_optimize(para, H_eff, beta_s_eff, opts);
    reward_ppo = smooth_reward(reward_ppo_raw, smooth_window);
    agent_ppo_best = agent_ppo;
    
    %% --- DDPG ---
    fprintf('\n--- DDPG (M=%d) ---\n', M);
    opts = drl_options;
    if ~isempty(agent_ddpg_best), opts.init_agent = agent_ddpg_best; end
    [Rx_ddpg, f_ddpg, CRB_ddpg, reward_ddpg_raw, agent_ddpg] = ...
        DDPG_optimize(para, H_eff, beta_s_eff, opts);
    reward_ddpg = smooth_reward(reward_ddpg_raw, smooth_window);
    agent_ddpg_best = agent_ddpg;

    %% Save results
    save_dir = sprintf('./results/ris_M%d', M);
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    save_path = fullfile(save_dir, 'results.mat');
    save(save_path, ...
        'M', 'para', 'Phi', 'H_eff', 'beta_s_eff', ...
        'reward_td3',  'reward_td3_raw',  'CRB_td3',  'agent_td3',  'Rx_td3',  'f_td3',  ...
        'reward_ppo',  'reward_ppo_raw',  'CRB_ppo',  'agent_ppo',  'Rx_ppo',  'f_ppo',  ...
        'reward_ddpg', 'reward_ddpg_raw', 'CRB_ddpg', 'agent_ddpg', 'Rx_ddpg', 'f_ddpg');
    
    fprintf('\n[M=%d] Saved results to %s\n', M, save_path);
    fprintf('CRB -> TD3: %.6f | PPO: %.6f | DDPG: %.6f\n', ...
        CRB_td3, CRB_ppo, CRB_ddpg);
end

fprintf('\n\n===  Sequential Retraining Complete!  ===\n');
fprintf('Run plot_crb_vs_ris.m to see the consistent results.\n');
