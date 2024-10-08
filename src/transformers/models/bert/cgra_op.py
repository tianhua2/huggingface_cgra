import torch, math

def get_minq_maxq(bits: int, sym: bool):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = torch.tensor(-maxq -1)
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)

    return minq, maxq

def asym_quantize(x: torch.Tensor, bits: int):
    minq, maxq = get_minq_maxq(bits=bits, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)
    scale = (((xmax - xmin)*0.9).clamp(min=1e-5) / maxq)
    zero = -xmin
    q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

    return q, scale, zero

def asym_dequantize(q, scale, zero):
    return q * scale - zero

def frac_mult(x, y, bw):
    x=(x*(2**(bw-1))).to(torch.int64)
    y=(y*(2**(bw-1))).to(torch.int64)

    ans = (x * y).to(torch.int64)

    result = (ans/(2**(bw-1))).to(torch.int64)
    return result/(2**(bw-1))

def frac_exp2(x, bw, term):
    result = torch.zeros_like(x)
    factorial = 1
    ln2 = torch.log(torch.tensor(2))
    ln2 = (ln2*(2**(bw-1))).to(torch.int)/(2**(bw-1))
    power = torch.ones_like(x)

    for n in range(term):
        result += power / factorial
        power = frac_mult(power, x, bw)
        power = frac_mult(power, ln2, bw)

        factorial *= (n + 1)

    return result

def custom_int_exp(x, bw, term):
    q, scale, zero = asym_quantize(x, bw)
    fp_x = asym_dequantize(q, scale, zero)

    input = fp_x*torch.tensor(1.44238)
    int_part = torch.floor(input)
    frac_part = input - int_part

    result = torch.pow(2, int_part)*frac_exp2(frac_part, bw, term)
    return result
  
def frac_add(x, y, bw):
    x=(x*(2**(bw-1))).to(torch.int64)
    y=(y*(2**(bw-1))).to(torch.int64)

    ans = x + y
    ans[ans >= 2 ** (2 * bw - 1)] = (2 ** (2 * bw - 1)) - 1
    result = ans.to(torch.int64)
    return result/(2**(bw-1))

def frac_div(x, y, bw):
    x=(x*(2**(bw-1))).to(torch.int64)
    y=(y*(2**(bw-1))).to(torch.int64)
    return x / y

def custom_int_softmax(x, bw, term):
    x_max = torch.max(x)
    x = x - x_max
    x_exp = custom_int_exp(x, bw, term)
    x_sum = torch.tensor(0)
    for x_i in x_exp:
        x_sum = frac_add(x_sum, x_i, bw)
    return frac_div(x_exp, x_sum, bw)
