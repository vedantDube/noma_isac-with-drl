%% plot_sinr_vs_ris.m
% Plots average SINR vs RIS elements for TD3, PPO, DDPG

clc; close all;
addpath('./functions');

M_values = [50, 100, 200, 300, 400];
num_reps = 5;

n_M = length(M_values);
n_users = 4; % from para_init

% Allocate storage for SINRs
SINR_td3 = zeros(n_M, n_users);
SINR_ppo = zeros(n_M, n_users);
SINR_ddpg = zeros(n_M, n_users);

for m_idx = 1:n_M
    M = M_values(m_idx);
    save_dir = sprintf('./results/ris_M%d', M);
    save_path = fullfile(save_dir, 'results.mat');
    
    if exist(save_path, 'file')
        load(save_path, 'para', 'H_eff', 'Rx_td3', 'f_td3', 'Rx_ppo', 'f_ppo', 'Rx_ddpg', 'f_ddpg');
        
        % Compute rates
        rate_td3 = rate_calculator(para, H_eff, Rx_td3, f_td3);
        rate_ppo = rate_calculator(para, H_eff, Rx_ppo, f_ppo);
        rate_ddpg = rate_calculator(para, H_eff, Rx_ddpg, f_ddpg);
        
        % Compute SINR = 2^rate - 1
        SINR_td3(m_idx, :) = 2.^rate_td3 - 1;
        SINR_ppo(m_idx, :) = 2.^rate_ppo - 1;
        SINR_ddpg(m_idx, :) = 2.^rate_ddpg - 1;
    else
        warning('File %s not found', save_path);
    end
end

% Average over users
avg_SINR_td3 = mean(SINR_td3, 2);
avg_SINR_ppo = mean(SINR_ppo, 2);
avg_SINR_ddpg = mean(SINR_ddpg, 2);

% Convert to dB
avg_SINR_td3_db = 10*log10(avg_SINR_td3);
avg_SINR_ppo_db = 10*log10(avg_SINR_ppo);
avg_SINR_ddpg_db = 10*log10(avg_SINR_ddpg);

% Plot
figure;
plot(M_values, avg_SINR_td3_db, '-o', 'LineWidth', 2, 'Color', [0.2 0.5 0.9], 'DisplayName', 'TD3');
hold on;
plot(M_values, avg_SINR_ppo_db, '-s', 'LineWidth', 2, 'Color', [0.9 0.3 0.2], 'DisplayName', 'PPO');
plot(M_values, avg_SINR_ddpg_db, '-^', 'LineWidth', 2, 'Color', [0.2 0.75 0.3], 'DisplayName', 'DDPG');
hold off;

xlabel('RIS Elements');
ylabel('Average SINR (dB)');
title('Average SINR vs RIS Elements');
legend('show');
grid on;

% Save the figure
saveas(gcf, 'sinr_vs_ris.png');
fprintf('Plot saved as sinr_vs_ris.png\n');