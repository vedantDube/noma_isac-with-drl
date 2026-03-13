function [s, a, r, s2, d, logp] = buf_sample(buf, n)
    idx = randperm(buf.cnt, n);
    s  = buf.s(:, idx);
    a  = buf.a(:, idx);
    r  = buf.r(idx);
    s2 = buf.s2(:, idx);
    d  = buf.d(idx);
    if isfield(buf, 'logp')
        logp = buf.logp(:, idx);
    else
        logp = [];
    end
end
