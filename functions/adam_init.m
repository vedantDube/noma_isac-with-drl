function opt = adam_init(net, lr)
    opt.lr    = lr;
    opt.beta1 = 0.9;
    opt.beta2 = 0.999;
    opt.t     = 0;
    opt.mW = cell(1, net.L);
    opt.vW = cell(1, net.L);
    opt.mb = cell(1, net.L);
    opt.vb = cell(1, net.L);
    for l = 1:net.L
        opt.mW{l} = zeros(size(net.W{l}));
        opt.vW{l} = zeros(size(net.W{l}));
        opt.mb{l} = zeros(size(net.b{l}));
        opt.vb{l} = zeros(size(net.b{l}));
    end
end
