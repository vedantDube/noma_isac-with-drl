%% plot_ris_results.m
% Loads all saved RIS sweep results and generates four comparison figures:
%
%   Figure 1 – Smoothed Reward vs Episode (3 subplots, one per algorithm)
%              Each subplot has 5 lines for M = 50/100/200/300/400
%
%   Figure 2 – Best CRB Trace vs RIS size (line plot)
%              3 lines (TD3, PPO, DDPG), one point per M value
%
%   Figure 3 – Algorithm comparison for a chosen M (reward vs episode)
%              Three lines: TD3, PPO, DDPG for M_compare
%
%   Figure 4 – Sum Rate vs RIS Element Count
%              Three lines: TD3, PPO, DDPG, one point per M value
%
% Usage: Run from project root after ris_sweep.m has completed.

clc; close all;
% first try to load aggregated results produced by ris_sweep.m
agg_file = './results/ris_aggregate.mat';
if exist(agg_file,'file')
    load(agg_file, 'M_values', ...
                   'CRB_mean_td3','CRB_std_td3', ...
                   'CRB_mean_ppo','CRB_std_ppo', ...
                   'CRB_mean_ddpg','CRB_std_ddpg', ...
                   'SumRates_td3','SumRates_ppo','SumRates_ddpg');
    use_agg = true;
else
    warning('Aggregated file %s not found; falling back to single-run loading.', agg_file);
    use_agg = false;
    M_values   = [50, 100, 200, 300, 400];
end

M_compare  = 200;   % chosen M for Figure 3 algorithm comparison
colors     = lines(length(M_values));   % distinct colours per M
algo_colors = [0.2 0.5 0.9;   % TD3  – blue
               0.9 0.3 0.2;   % PPO  – red
               0.2 0.75 0.3]; % DDPG – green

%% Pre-allocate storage
n_M  = length(M_values);
if ~use_agg
    CRBs     = zeros(n_M, 3);   % columns: TD3, PPO, DDPG
    SumRates = zeros(n_M, 3);   % columns: TD3, PPO, DDPG
    rewards_td3  = cell(n_M, 1);
    rewards_ppo  = cell(n_M, 1);
    rewards_ddpg = cell(n_M, 1);
end

%% Load results, re-smooth, and compute sum rates
addpath('./functions');
smooth_win = 200;   % large window for very clean curves (double-pass applied below)

if ~use_agg
    for i = 1:n_M
        M = M_values(i);
        fpath = sprintf('./results/ris_M%d/results.mat', M);
        if ~exist(fpath, 'file')
            error('Result file not found: %s\nRun ris_sweep.m first.', fpath);
        end
        d = load(fpath);
        % Double-pass smoothing from raw data for very clean curves
        rewards_td3{i}  = smooth_reward(smooth_reward(d.reward_td3_raw(:),  smooth_win), smooth_win);
        rewards_ppo{i}  = smooth_reward(smooth_reward(d.reward_ppo_raw(:),  smooth_win), smooth_win);
        rewards_ddpg{i} = smooth_reward(smooth_reward(d.reward_ddpg_raw(:), smooth_win), smooth_win);
        % Handle empty CRB (if a run failed to find feasible solution)
        CRBs(i, 1) = safeVal(d.CRB_td3);
        CRBs(i, 2) = safeVal(d.CRB_ppo);
        CRBs(i, 3) = safeVal(d.CRB_ddpg);

        % Load the exact effective channel that agents were trained on
        para_m = d.para;
        if isfield(d, 'H_eff')
            H_eff = d.H_eff;   % saved by ris_sweep.m  (preferred)
        else
            % Legacy fallback: try to re-derive (may differ due to RNG state)
            warning('H_eff not found in %s. Re-running ris_sweep.m is recommended.', fpath);
            para_base_fb = para_init();
            rng(42);
            [H_dir_fb, ~, bs_fb, r_users_fb, th_users_fb, ~, ~] = generate_channel(para_base_fb);
            [H_eff, ~, ~] = generate_ris_channel(para_m, H_dir_fb, bs_fb, r_users_fb, th_users_fb);
        end

        % Compute sum rate = sum of per-user log2(1+SINR) for each algorithm
        rate_td3  = rate_calculator(para_m, H_eff, d.Rx_td3,  d.f_td3);
        rate_ppo  = rate_calculator(para_m, H_eff, d.Rx_ppo,  d.f_ppo);
        rate_ddpg = rate_calculator(para_m, H_eff, d.Rx_ddpg, d.f_ddpg);
        SumRates(i, 1) = sum(rate_td3);
        SumRates(i, 2) = sum(rate_ppo);
        SumRates(i, 3) = sum(rate_ddpg);
    end
end

%% ============================================================
%  Figure 1 – Reward vs Episode per algorithm (5 RIS sizes)
%% ============================================================
figure('Name', 'Smoothed Reward vs Episode (per Algorithm)', ...
       'Position', [50 50 1300 420]);

algo_names = {'TD3', 'PPO', 'DDPG'};
reward_data = {rewards_td3, rewards_ppo, rewards_ddpg};

for a = 1:3
    subplot(1, 3, a);
    hold on;
    for i = 1:n_M
        rv = reward_data{a}{i};
        plot(1:length(rv), rv, 'LineWidth', 1.8, 'Color', colors(i,:));
    end
    hold off;
    xlabel('Episode', 'FontSize', 11);
    ylabel('Smoothed Reward', 'FontSize', 11);
    title(sprintf('%s – Reward vs Episode', algo_names{a}), 'FontSize', 12);
    legend(arrayfun(@(m) sprintf('M=%d', m), M_values, 'UniformOutput', false), ...
           'Location', 'best', 'FontSize', 9);
    grid on; box on;
end
sgtitle('Smoothed Training Reward vs Episode for Different RIS Sizes', ...
        'FontSize', 13, 'FontWeight', 'bold');

%% ============================================================
%  Figure 2 – Best CRB vs RIS Size (line or errorbar)
%% ============================================================
figure('Name', 'CRB vs RIS Size', 'Position', [50 530 700 420]);

hold on;
if use_agg
    errorbar(M_values, CRB_mean_td3, CRB_std_td3, '-o', 'LineWidth', 2.0, 'MarkerSize', 8, ...
         'Color', algo_colors(1,:), 'MarkerFaceColor', algo_colors(1,:));
    errorbar(M_values, CRB_mean_ppo, CRB_std_ppo, '-s', 'LineWidth', 2.0, 'MarkerSize', 8, ...
         'Color', algo_colors(2,:), 'MarkerFaceColor', algo_colors(2,:));
    errorbar(M_values, CRB_mean_ddpg, CRB_std_ddpg, '-^', 'LineWidth', 2.0, 'MarkerSize', 8, ...
         'Color', algo_colors(3,:), 'MarkerFaceColor', algo_colors(3,:));
else
    for a = 1:3
        plot(M_values, CRBs(:, a), '-o', 'LineWidth', 2.0, 'MarkerSize', 8, ...
             'Color', algo_colors(a,:), 'MarkerFaceColor', algo_colors(a,:));
    end
end
hold off;
xlabel('RIS Size (M)', 'FontSize', 12);
ylabel('Best CRB Trace', 'FontSize', 12);
title('Best CRB Trace vs RIS Element Count', 'FontSize', 13, 'FontWeight', 'bold');
legend({'TD3', 'PPO', 'DDPG'}, 'Location', 'northeast', 'FontSize', 11);
grid on; box on;

%% ============================================================
%  Figure 3 – Algorithm comparison for M = M_compare
%% ============================================================
m_idx_sel = find(M_values == M_compare);
if isempty(m_idx_sel)
    warning('M_compare=%d not in M_values. Using M=%d instead.', ...
        M_compare, M_values(round(n_M/2)));
    m_idx_sel = round(n_M/2);
end

figure('Name', sprintf('Algorithm Comparison (M=%d)', M_values(m_idx_sel)), ...
       'Position', [800 530 700 420]);
hold on;
rv_td3  = rewards_td3{m_idx_sel};
rv_ppo  = rewards_ppo{m_idx_sel};
rv_ddpg = rewards_ddpg{m_idx_sel};
plot(1:length(rv_td3),  rv_td3,  'LineWidth', 2.0, 'Color', algo_colors(1,:));
plot(1:length(rv_ppo),  rv_ppo,  'LineWidth', 2.0, 'Color', algo_colors(2,:));
plot(1:length(rv_ddpg), rv_ddpg, 'LineWidth', 2.0, 'Color', algo_colors(3,:));
hold off;
legend({'TD3', 'PPO', 'DDPG'}, 'Location', 'best', 'FontSize', 11);
xlabel('Episode', 'FontSize', 12);
ylabel('Smoothed Reward', 'FontSize', 12);
title(sprintf('Algorithm Comparison  |  RIS M = %d Elements', ...
    M_values(m_idx_sel)), 'FontSize', 13, 'FontWeight', 'bold');
grid on; box on;

%% ============================================================
%  Figure 4 – Sum Rate vs RIS Element Count
%% ============================================================
figure('Name', 'Sum Rate vs RIS Element Count', 'Position', [500 50 750 480]);

hold on;
for a = 1:3
    plot(M_values, SumRates(:, a), '-s', ...
        'LineWidth', 2.2, ...
        'MarkerSize', 9, ...
        'Color', algo_colors(a,:), ...
        'MarkerFaceColor', algo_colors(a,:));
end
hold off;

xlabel('Number of RIS Elements (M)', 'FontSize', 13);
ylabel('Sum Rate (bps/Hz)',          'FontSize', 13);
title('Sum Rate vs RIS Element Count', 'FontSize', 14, 'FontWeight', 'bold');
legend({'TD3', 'PPO', 'DDPG'}, 'Location', 'best', 'FontSize', 12);
xticks(M_values);
grid on; box on;
set(gca, 'FontSize', 11);

% Print table to console
fprintf('\n== Sum Rate vs RIS Element Count ==\n');
if use_agg
    % compute mean and std across reps
    mean_td3 = mean(SumRates_td3,1);
    std_td3  = std(SumRates_td3,0,1);
    mean_ppo = mean(SumRates_ppo,1);
    std_ppo  = std(SumRates_ppo,0,1);
    mean_ddpg = mean(SumRates_ddpg,1);
    std_ddpg  = std(SumRates_ddpg,0,1);
    fprintf('%-6s  %15s  %15s  %15s\n', 'M', 'TD3 (mean±std)', 'PPO (mean±std)', 'DDPG (mean±std)');
    for i = 1:n_M
        fprintf('%-6d  %10.4f±%5.4f  %10.4f±%5.4f  %10.4f±%5.4f\n', ...
            M_values(i), mean_td3(i), std_td3(i), mean_ppo(i), std_ppo(i), mean_ddpg(i), std_ddpg(i));
    end
else
    fprintf('%-6s  %10s  %10s  %10s\n', 'M', 'TD3', 'PPO', 'DDPG');
    for i = 1:n_M
        fprintf('%-6d  %10.4f  %10.4f  %10.4f\n', ...
            M_values(i), SumRates(i,1), SumRates(i,2), SumRates(i,3));
    end
end

fprintf('\nDone! Figures generated for M = %s\n', ...
    strjoin(arrayfun(@(m)sprintf('%d',m), M_values,'UniformOutput',false), ', '));

%% ---- Helper ----
function v = safeVal(x)
% Return scalar value even if x is empty/NaN
    if isempty(x) || (isscalar(x) && isnan(x))
        v = NaN;
    else
        v = double(x);
    end
end
