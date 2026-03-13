clc
clear all
close all

addpath('./functions');
cvx_solver mosek


para = para_init();

[H, G, beta_s, r, theta, r_s, theta_s] = generate_channel(para);

% Sort users by channel gain in ascending order (NOMA)
channel_gain = zeros(para.K, 1);
for k = 1:para.K
    channel_gain(k) = norm(H(:,k))^2;
end
[channel_gain_sorted, idx] = sort(channel_gain, 'ascend');
H = H(:, idx);
r = r(idx);
theta = theta(idx);
disp('Sorted channel gains ||h_k||^2 (ascending):');
disp(channel_gain_sorted.');

scale = 1e2; % if the optimization fails, try to adjust this scale factor

% Optimize the transmit waveform
[Rx, f] = SDR_fully_digital(para, H, beta_s, scale);

% Calculate the communication rates for all users
[rate] = rate_calculator(para, H, Rx, f);

% Calculate the CRB matrix
[J_11, J_12, J_22] = FIM(para, Rx, beta_s, scale);
scale = scale*para.noise/para.T;
J_11 = J_11/scale;
J_12 = J_12/scale;
J_22 = J_22/scale;
CRB = real(inv(J_11 - J_12*inv(J_22)*J_12.'));

% MUSIC algorithm
[spectrum, X, Y] = MUSIC_estimation(para, Rx, f, G);

figure; colormap jet;
mesh(X,Y,10*log10(spectrum)); 
view([60,60]);
xlim([0,40]);
ylim([0,40]);
xlabel('x (m)'); ylabel('y (m)'); zlabel('Spectrum (dB)');
title('SDR MUSIC Spectrum');


%% ======== DRL (TD3) Optimization for Robust Sensing ========
fprintf('\n========== DRL (TD3) Optimization ==========\n');

% Training options (adjust num_episodes for longer training)
drl_options.num_episodes  = 1000;  % number of training episodes (increased)
drl_options.hidden_dim    = 256;   % hidden layer size
drl_options.warmup_steps  = 500;   % random exploration steps before training (reduced)
drl_options.channel_noise = 0.01;  % channel noise for robustness (reduced)
drl_options.explore_noise = 0.3;   % initial exploration noise std (annealed in TD3)
drl_options.batch_size    = 128;   % mini-batch size

% Run TD3 optimizer
[Rx_drl, f_drl, CRB_drl_trace, reward_hist] = TD3_optimize(para, H, beta_s, drl_options);

% Calculate DRL communication rates
[rate_drl] = rate_calculator(para, H, Rx_drl, f_drl);

%% ======== Comparison: SDR vs DRL ========
fprintf('\n========== Results Comparison ==========\n');
fprintf('SDR CRB Trace:  %.6f\n', trace(CRB));
fprintf('DRL CRB Trace:  %.6f\n', CRB_drl_trace);
fprintf('\nSDR Communication Rates:\n');
for k = 1:para.K
    fprintf('  User %d: %.4f bps/Hz\n', k, rate(k));
end
fprintf('\nDRL Communication Rates:\n');
for k = 1:para.K
    fprintf('  User %d: %.4f bps/Hz\n', k, rate_drl(k));
end

% Plot DRL training reward curve
figure;
plot(1:length(reward_hist), reward_hist, 'b-', 'LineWidth', 1.5);
xlabel('Episode'); ylabel('Total Reward');
title('TD3 Training Progress');
grid on;

% Plot DRL MUSIC spectrum for comparison
[spectrum_drl, X_drl, Y_drl] = MUSIC_estimation(para, Rx_drl, f_drl, G);

figure; colormap jet;
mesh(X_drl, Y_drl, 10*log10(spectrum_drl));
view([60,60]);
xlim([0,40]);
ylim([0,40]);
xlabel('x (m)'); ylabel('y (m)'); zlabel('Spectrum (dB)');
title('DRL (TD3) MUSIC Spectrum');

% CRB comparison bar chart
figure;
bar([trace(CRB), CRB_drl_trace]);
set(gca, 'XTickLabel', {'SDR', 'DRL (TD3)'});
ylabel('CRB Trace');
title('CRB Comparison: SDR vs DRL');
grid on;

%% ======== PPO and DDPG Optimization & Comparison ========
drl_options.max_episodes = drl_options.num_episodes;
[~, ~, ~, ppo_reward_hist] = PPO_optimize(para, H, beta_s, drl_options);
[~, ~, ~, ddpg_reward_hist] = DDPG_optimize(para, H, beta_s, drl_options);

% Plot reward vs episode for TD3, PPO, DDPG
figure;
hold on;
plot(1:length(reward_hist), reward_hist, 'b-', 'LineWidth', 1.5);
plot(1:length(ppo_reward_hist), ppo_reward_hist, 'r-', 'LineWidth', 1.5);
plot(1:length(ddpg_reward_hist), ddpg_reward_hist, 'g-', 'LineWidth', 1.5);
hold off;
legend('TD3', 'PPO', 'DDPG');
xlabel('Episode'); ylabel('Total Reward');
title('Reward vs Episode: TD3 vs PPO vs DDPG');
grid on;



