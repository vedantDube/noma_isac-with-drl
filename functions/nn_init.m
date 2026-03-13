function net = nn_init(dims)
% Initialize fully-connected network with Xavier/He initialization
% dims = [input_dim, hidden1, hidden2, ..., output_dim]
    L = length(dims) - 1;
    net.W = cell(1, L);
    net.b = cell(1, L);
    net.L = L;
    for l = 1:L
        % He initialization for ReLU
        net.W{l} = randn(dims(l+1), dims(l)) * sqrt(2 / dims(l));
        net.b{l} = zeros(dims(l+1), 1);
    end
end
