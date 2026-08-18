import math

def transpose(matrix):
    return [list(col) for col in zip(*matrix)]

def matmul(A, B):
    # mult A by B
    assert len(A[1]) == len(B[0]), \
        "second dim of first matrix must equal first dim of second" 

    out = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    # iterate through rows
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                out[i][j] += A[i][k] * B[k][j]

    return out

def add_bias(A, b):
    return [[x + bi for x, bi in zip(row, b)] for row in A]

def linear(X, W, b):
    return add_bias(matmul(X, W), b)

def softmax(row):
    m = max(row)
    exps = [math.exp(num- m) for num in row]
    total = sum(exps)
    return [exp / total for exp in exps]

def add(A, B):
    out = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            out[i][j] = A[i][j] + B[i][j]

    return out 

def relu(x):
    return max(0, x)

def layernorm(X, gamma, beta, eps=1e-5):
    out = []
    for row in X:
        n = len(row)
        mean = sum(row) / n
        var = sum((v - mean)**2 for v in row) / n
        denom = math.sqrt(var +  eps)
        out.append(
            [g * (v - mean) / denom + b 
            for v, b, g in zip(row, gamma, beta)]
        )

def split_head(X, h, d_head):
    low, high = h * d_head, (h + 1) * d_head
    return [row[low:high] for row in X]


def scaled_dot_product_attention(Q, K, V):
    S, d_head = len(Q), len(Q[0])
    attn_scores = matmul(Q, transpose(K)) 
    scale = math.sqrt(d_head)
    attn_scores = [[num / scale for num in row] for row in attn_scores]
    
    attn_probs = []
    for i, row in enumerate(attn_scores):
        masked = [row[j] if j <= i else float("-inf") for j in range(S)]
        attn_probs.append(softmax(masked))

    return matmul(attn_probs, V)


def multi_head_attention(X, p, n_heads):
    d_model = len(X[0])
    d_head = d_model // n_heads

    Q = linear(X, p["Wq"], p["bq"])
    K = linear(X, p["Wk"], p["bk"])
    V = linear(X, p["Wv"], p["bv"])

    attn = [scaled_dot_product_attention(split_head(Q, h, d_head),
                                         split_head(K, h, d_head),
                                         split_head(V, h, d_head))
            for h in range(n_heads)]

    combine_heads = [sum((head[t] for head in attn), []) for t in range(len(X))] 
    return linear(combine_heads, p["Wo"], p["Wb"])


def feed_forward(X, p):
    H = linear(X, p["W1"], p["b1"])
    H = [[relu(v) for v in row] for row in H]
    return linear(H, p["W2"], p["b2"])



if __name__ == "__main__":
    matrix1 = [[0, 1], [2, 3]]
    matrix2 = [[0, 1], [2, 3]]
    matrix3 = [[0, 1], [2, 3]]
    print(scaled_dot_product_attention(matrix1, matrix2, matrix3))
