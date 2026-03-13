function [next_state, reward, done, env] = drl_step(env, action, para)
%DRL Environment Step Function for Near-Field ISAC
%   [next_state, reward, done, env] = drl_step(env, action, para)
%
% Executes one environment step:
%   1. Converts action to MMSE-based beamforming + power/sensing allocation
%   2. Computes CRB (Cramer-Rao Bound) for sensing performance
%   3. Computes communication rates (NOMA with SIC)
%   4. Returns reward that balances CRB minimization and rate constraints
%
% Inputs:
%   env    - environment struct from drl_reset
%   action - action vector [K+1 x 1], each value in [-1, 1]
%            action(1:K)   -> power allocation weights per user
%            action(K+1)   -> sensing power fraction
%   para   - system parameters
%
% Outputs:
%   next_state - next state vector
%   reward     - scalar reward signal
%   done       - true if episode is finished
%   env        - updated environment struct

    H = env.H;
    beta_s = env.beta_s;
    K = para.K;
    N = para.N;

    env.step_count = env.step_count + 1;

    %% Parse action (all values in [-1, 1] from tanh output)
    % Power allocation weights -> mapped to [0.1, 1]
    power_raw = (action(1:K) + 1) / 2;
    power_raw = max(power_raw, 0.1);

    % Sensing power fraction -> mapped to [0.05, 0.35]
    sensing_frac = 0.05 + 0.30 * (action(K+1) + 1) / 2;
    %% Maximum Ratio Transmission (MRT) Beamforming for NOMA
    total_comm_power = (1 - sensing_frac) * para.Pt;
    power_alloc = power_raw / sum(power_raw) * total_comm_power;

    % MRT maximizes array gain. The interference caused by spatial 
    % correlation (due to the RIS) will be handled by the NOMA SIC block below.
    f = zeros(N, K);
    for k = 1:K
        wk = H(:,k);  % MRT simply matches the user's effective channel
        f(:,k) = sqrt(power_alloc(k)) * wk / norm(wk);
    end
    
    %% Construct transmit covariance matrix Rx = f*f' + sensing component
    sensing_power = sensing_frac * para.Pt;
    a_s = beamfocusing(para, para.r_s, para.theta_s);
    S_sense = sensing_power * (a_s * a_s') / real(a_s' * a_s);
    Rx = f * f' + S_sense;
    Rx = (Rx + Rx') / 2;   % ensure Hermitian

    % Enforce total power constraint
    rx_power = real(trace(Rx));
    if rx_power > para.Pt
        sf = sqrt(para.Pt / rx_power);
        Rx = Rx * (para.Pt / rx_power);
        f = f * sf;
    end

    %% Compute CRB from Fisher Information Matrix
    scale = 1e2;
    [J_11, J_12, J_22] = FIM(para, Rx, beta_s, scale);
    scale_factor = scale * para.noise / para.T;
    J_11 = J_11 / scale_factor;
    J_12 = J_12 / scale_factor;
    J_22 = J_22 / scale_factor;

    eigvals_J22 = eig(J_22);
    if all(real(eigvals_J22) > 1e-10)
        Schur = J_11 - J_12 * (J_22 \ J_12.');
        eigvals_S = eig(Schur);
        if all(real(eigvals_S) > 1e-10)
            CRB_mat = real(inv(Schur));
            crb_trace = trace(CRB_mat);
        else
            crb_trace = 1e3;
        end
    else
        crb_trace = 1e3;
    end

    %% Compute communication rates (NOMA SIC - ascending channel gain order)
    % User k cancels weaker users 1..k-1, sees interference from k+1..K
    % Sensing signal is known/deterministic -> cancelable at receivers
    rate = zeros(K, 1);
    for k = 1:K
        hk = H(:,k);
        fk = f(:,k);
        sig = abs(hk.' * fk)^2;
        interference = 0;
        for i = k+1:K
            interference = interference + abs(hk.' * f(:,i))^2;
        end
        rate(k) = log2(1 + sig / (interference + 1));
    end

    %% Compute reward (balanced CRB + rate)
    crb_ref = env.crb_ref;
    crb_reward = -log(max(crb_trace, 1e-10) / max(crb_ref, 1e-10));
    crb_reward = max(min(crb_reward, 5), -5);

    % Rate component: per-user bonus/penalty
    rate_bonus = 0;
    rate_deficit = 0;
    for k = 1:K
        if rate(k) >= para.Rmin
            rate_bonus = rate_bonus + 1;
        else
            % Softer penalty for weak users missing target (reduced from 2.0 to 0.5)
            rate_deficit = rate_deficit + 0.5 * (rate(k) - para.Rmin) / para.Rmin;
        end
    end

    % CRB reward only kicks in meaningfully when most rates are met
    feas_ratio = rate_bonus / K;  % 0 to 1
    reward = (1 + feas_ratio) * crb_reward + rate_bonus + rate_deficit;

    %% Track best feasible solution in this episode
    env.crb_trace = crb_trace;
    all_rates_met = all(rate >= para.Rmin * 0.8);  % 80% tolerance for tracking
    prev_rates_met = ~isempty(env.best_rate) && all(env.best_rate >= para.Rmin * 0.8);
    if all_rates_met && crb_trace < env.best_CRB
        % Feasible and better CRB
        env.best_reward = reward;
        env.best_Rx = Rx;
        env.best_f = f;
        env.best_CRB = crb_trace;
        env.best_rate = rate;
    elseif all_rates_met && ~prev_rates_met
        % First feasible solution (prev was infeasible)
        env.best_reward = reward;
        env.best_Rx = Rx;
        env.best_f = f;
        env.best_CRB = crb_trace;
        env.best_rate = rate;
    elseif ~prev_rates_met && crb_trace < env.best_CRB
        % No feasible solution yet, track best CRB anyway
        env.best_reward = reward;
        env.best_Rx = Rx;
        env.best_f = f;
        env.best_CRB = crb_trace;
        env.best_rate = rate;
    end

    %% Check episode termination
    done = (env.step_count >= env.max_steps);

    %% Compute next state
    % [normalized rates, normalized CRB, sensing fraction, step progress]
    rate_n = rate / max(para.Rmin, 1);         % normalized by target rate
    crb_n = min(crb_trace / crb_ref, 10);      % normalized by reference CRB
    step_progress = env.step_count / env.max_steps;

    next_state = [rate_n; crb_n; sensing_frac; step_progress];
end
