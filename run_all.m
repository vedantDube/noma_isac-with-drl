% run_all.m
% Convenience script that executes the main examples in the repository.
% This lets the user simply type "run_all" in MATLAB and have the
% SDR baseline, the DRL training, the RIS sweep, and the plotting
% performed automatically.
%
% Usage: open MATLAB, cd to project root, and run:
%    run_all
%
% Requires CVX/MOSEK on the path for the SDR portion. The RIS sweep
% may take several minutes depending on `drl_options.num_episodes`.

clc; close all;

% make sure function directory is available
addpath('./functions');

fprintf('=== Running SDR + DRL example from main.m ===\n');
try
    main;
catch ME
    warning('main.m execution failed: %s', ME.message);
end

fprintf('\n=== Performing RIS sweep (multiple trials, deterministic) ===\n');
try
    ris_sweep;
catch ME
    warning('ris_sweep.m failed: %s', ME.message);
end

fprintf('\n=== Generating comparison plots ===\n');
try
    plot_ris_results;
catch ME
    warning('plot_ris_results.m failed: %s', ME.message);
end

fprintf('\n=== All scripts have finished.\nResults saved in the ./results directory.\n');
