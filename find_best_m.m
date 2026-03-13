addpath('./functions');
M_list = [50, 100, 200, 300, 400];
fprintf('M, TD3_CRB, PPO_CRB, DDPG_CRB\n');
for M = M_list
    fpath = sprintf('./results/ris_M%d/results.mat', M);
    if exist(fpath, 'file')
        d = load(fpath);
        
        % Check if fields exist
        if isfield(d, 'CRB_td3')
            td3 = d.CRB_td3;
        else
            td3 = NaN;
        end
        if isfield(d, 'CRB_ppo')
            ppo = d.CRB_ppo;
        else
            ppo = NaN;
        end
        if isfield(d, 'CRB_ddpg')
            ddpg = d.CRB_ddpg;
        else
            ddpg = NaN;
        end
        
        fprintf('%d, %.6f, %.6f, %.6f\n', M, td3, ppo, ddpg);
    else
        fprintf('%d, File Not Found: %s\n', M, fpath);
    end
end
