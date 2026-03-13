function buf = buf_store(buf, s, a, r, s2, d, logp)
    idx = mod(buf.ptr, buf.cap) + 1;
    buf.s(:, idx)  = s;
    buf.a(:, idx)  = a;
    buf.r(idx)     = r;
    buf.s2(:, idx) = s2;
    buf.d(idx)     = d;
    if nargin > 6
        buf.logp(:, idx) = logp;
    end
    buf.ptr = buf.ptr + 1;
    buf.cnt = min(buf.cnt + 1, buf.cap);
end
