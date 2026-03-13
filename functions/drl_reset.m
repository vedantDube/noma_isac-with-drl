function [state, env] = drl_reset(para, H_in, beta_s_in)
%DRL Environment Reset Function for Near-Field ISAC
%   [state, env] = drl_reset(para)
%   [state, env] = drl_reset(para, H, beta_s)
%
% Resets the RL environment for CRB minimization.
% State = [rates(K), normalized_CRB, sensing_frac, step_progress]
% State dimension: K + 3
%
% Inputs:
%   para      - system parameters (from para_init)
%   H_in      - (optional) communication channel matrix [N x K]
%   beta_s_in - (optional) complex sensing channel gain
%
% Outputs:
%   state - initial state vector [K+3, 1]
%   env   - environment struct with channel info and episode tracking

    K = para.K;
    N = para.N;

    if nargin >= 3 && ~isempty(H_in)
        H = H_in;
        beta_s = beta_s_in;
    else
        % Generate new channel realization
        [H, ~, beta_s] = generate_channel(para);

        % Sort users by channel gain in ascending order (NOMA SIC ordering)
        channel_gain = zeros(K, 1);
        for k = 1:K
            channel_gain(k) = norm(H(:,k))^2;
        end
        [~, idx] = sort(channel_gain, 'ascend');
        H = H(:, idx);
    end

    % Store environment state
    env.H = H;
    env.beta_s = beta_s;
    env.step_count = 0;
    env.max_steps = 50;       % steps per episode
    env.best_reward = -inf;
    env.best_Rx = [];
    env.best_f = [];
    env.best_CRB = inf;
    env.best_rate = zeros(K, 1);
    env.crb_trace = 0;

    % Compute a reference CRB using equal power uniform allocation
    % This gives a baseline for reward normalization
    a_s = beamfocusing(para, para.r_s, para.theta_s);
    Rx_ref = (para.Pt / N) * eye(N);
    scale = 1e2;
    [J_11, J_12, J_22] = FIM(para, Rx_ref, beta_s, scale);
    sf = scale * para.noise / para.T;
    J_11 = J_11 / sf; J_12 = J_12 / sf; J_22 = J_22 / sf;
    eJ22 = eig(J_22);
    if all(real(eJ22) > 1e-10)
        Schur = J_11 - J_12 * (J_22 \ J_12.');
        eS = eig(Schur);
        if all(real(eS) > 1e-10)
            env.crb_ref = trace(real(inv(Schur)));
        else
            env.crb_ref = 1.0;
        end
    else
        env.crb_ref = 1.0;
    end

    % Initial state: zeros for rates, 1 for normalized CRB, 0.2 sensing frac, 0 progress
    state = [zeros(K, 1); 1.0; 0.2; 0];
end
