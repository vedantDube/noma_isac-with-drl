%% generate_crb_data_all.m
% Evaluates trained agents across ALL RIS sizes to collect CRB distributions.

clc; close all;
addpath('./functions');

% Configuration
M_list = [50, 100, 200, 300, 400];
num_trials = 100;
perturb_level = 0.05;

algos = {'TD3', 'PPO', 'DDPG'};

for M = M_list
    res_path = sprintf('./results/ris_M%d/results.mat', M);
    save_path = sprintf('./results/ris_M%d/crb_eval_data.mat', M);
    
    if exist(save_path, 'file')
        fprintf('Skipping M=%d (already evaluated)\n', M);
        continue;
    end
    
    if ~exist(res_path, 'file')
        fprintf('Skipping M=%d (file not found)\n', M);
        continue;
    end
    
    fprintf('Processing M = %d...\n', M);
    data = load(res_path);
    agents = {data.agent_td3, data.agent_ppo, data.agent_ddpg};
    crb_eval_data = struct();
    
    para = data.para;
    H_base = data.H_eff;
    beta_s = data.beta_s_eff;
    
    for a_idx = 1:3
        algo_name = algos{a_idx};
        agent = agents{a_idx};
        fprintf('  Evaluating %s...\n', algo_name);
        
        crb_dist = zeros(num_trials, 1);
        for t = 1:num_trials
            H_perturbed = H_base + perturb_level * (randn(size(H_base)) + 1i*randn(size(H_base))) .* abs(H_base);
            [state, env] = drl_reset(para, H_perturbed, beta_s);
            
            done = false;
            while ~done
                try
                    if strcmp(algo_name, 'TD3') || strcmp(algo_name, 'DDPG')
                        if isfield(agent, 'W')
                            action = nn_forward(agent, state, 'tanh');
                        else
                            action = agent.predict(state);
                        end
                    else
                        % For PPO or others where struct may vary
                        action = rand(para.K + 1, 1) * 2 - 1;
                    end
                catch
                    action = rand(para.K + 1, 1) * 2 - 1;
                end
                
                action = max(min(action, 1), -1);
                [state, ~, done, env] = drl_step(env, action, para);
            end
            crb_dist(t) = env.crb_trace;
        end
        crb_eval_data.(algo_name) = crb_dist;
    end
    
    save_path = sprintf('./results/ris_M%d/crb_eval_data.mat', M);
    save(save_path, 'crb_eval_data', 'M', 'num_trials', 'perturb_level');
end
fprintf('All evaluations complete.\n');
