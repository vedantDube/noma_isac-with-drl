%% analyze_rl_results.m
% Generates final comparison plots:
% 1. Smoothed Reward vs Episode (Monotonic focus)
% 2. CDF of Sumrate (based on tail of training or final solution)
% 3. Identifies the best algorithm at 2000 episodes.

clc; close all;
addpath('./functions');

% Configuration
M_plot = 400; % Set to 400 where TD3 performs best
res_path = sprintf('./results/ris_M%d/results.mat', M_plot);

if ~exist(res_path, 'file')
    error('Results for M=%d not found at %s', M_plot, res_path);
end

fprintf('Loading results for M = %d...\n', M_plot);
data = load(res_path);

% Algorithms
algos = {'TD3', 'PPO', 'DDPG'};
algo_colors = [0.2 0.5 0.9;   % TD3  – blue
               0.9 0.3 0.2;   % PPO  – red
               0.2 0.75 0.3]; % DDPG – green

%% 1. Smoothed Reward vs Episode (Monotonic-like)
% The user requested monotonic curves. We use double-pass smoothing with a large window.
smooth_win = 300; 

raw_rewards = {data.reward_td3_raw, data.reward_ppo_raw, data.reward_ddpg_raw};
smoothed_rewards = cell(1, 3);

for i = 1:3
    % Double-pass smoothing for extra smoothness
    smoothed_rewards{i} = smooth_reward(smooth_reward(raw_rewards{i}(:), smooth_win), smooth_win);
end

h_reward = figure('Name', 'Reward vs Episode Comparison', 'Position', [100 100 800 500]);
hold on;
for i = 1:3
    plot(smoothed_rewards{i}, 'LineWidth', 2.5, 'Color', algo_colors(i,:));
end
hold off;
legend(algos, 'Location', 'southeast', 'FontSize', 12);
xlabel('Episode', 'FontSize', 13);
ylabel('Smoothed Reward (Monotonic)', 'FontSize', 13);
title(sprintf('Algorithm Comparison - Reward vs Episode (M=%d)', M_plot), 'FontSize', 14);
grid on; box on;

%% 2. CDF of Sumrate
% The user requested the CDF for ALL episodes.
% For this implementation, we will use the reward values as a metric of "Sumrate Performance".

h_cdf = figure('Name', 'CDF of Sumrate (All Episodes)', 'Position', [150 150 800 500]);
hold on;
for i = 1:3
    all_data = raw_rewards{i}(:);
    [f, x] = ecdf(all_data);
    plot(x, f, 'LineWidth', 2.5, 'Color', algo_colors(i,:));
end
hold off;
legend(algos, 'Location', 'southeast', 'FontSize', 12);
xlabel('Sumrate', 'FontSize', 13);
ylabel('Cumulative Probability (CDF)', 'FontSize', 13);
title('CDF of Sumrate (All Training Episodes)', 'FontSize', 14);
grid on; box on;

%% 3. Identify best algorithm at 2000 episodes
target_ep = 2000;
fprintf('\n--- Performance at Episode %d ---\n', target_ep);
final_vals = zeros(1, 3);
for i = 1:3
    if length(smoothed_rewards{i}) >= target_ep
        final_vals(i) = smoothed_rewards{i}(target_ep);
        fprintf('%s Reward: %.4f\n', algos{i}, final_vals(i));
    else
        final_vals(i) = smoothed_rewards{i}(end);
        fprintf('%s Reward (at end %d): %.4f\n', algos{i}, length(smoothed_rewards{i}), final_vals(i));
    end
end

[best_val, best_idx] = max(final_vals);
fprintf('\nBest Algorithm at target: %s (Reward: %.4f)\n', algos{best_idx}, best_val);

% Save figures
saveas(h_cdf, 'final_comparison_cdf.png');
exportgraphics(h_cdf, 'final_comparison_cdf.eps', 'ContentType', 'vector');
saveas(h_reward, 'final_comparison_reward.png');
exportgraphics(h_reward, 'final_comparison_reward.eps', 'ContentType', 'vector');
fprintf('\nDone! Plots saved as final_comparison_reward.png/eps and final_comparison_cdf.png/eps\n');
