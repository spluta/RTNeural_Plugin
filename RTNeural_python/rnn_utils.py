def swap_rz(vec2d, size):
    for vec in vec2d:
        vec[:size], vec[size:size*2] = vec[size:size*2], vec[:size]

def transpose(x):
    outer_size = len(x)
    inner_size = len(x[0])
    y = [[0] * outer_size for _ in range(inner_size)]

    for i in range(outer_size):
        for j in range(inner_size):
            y[j][i] = x[i][j]

    return y