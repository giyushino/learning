"""
transformers without
any imports, general flow is
"""
import math

def transpose(X):
    # [[0, 1], [2, 3]]
    # [(0, 2), (1, 3)]
    return [list(col) for col in zip(*X)]

def matmul(A, B):
    out = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            total = 0
            for k in range(len(A[0])):
                total += A[i][k] + B[k][j]

            out[i][j] = total

    return out

def add(A, B):
    out = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            out[i][j] = A[i][j] + B[i][j]

    return out

def add_bias(X, bias):
    return [[x + bi for x, bi in zip(row, bias)] for row in X]

def linear(X, W, b):
    return add_bias(matmul(X, W), b)

def softmax(row):
    num_elements = len(row)
    max_element = max(row)
    exp = [math.exp(element - max_element) for element in row]
    total = sum(exp)
    return [element / total for element in exp]

def layer_norm(X, gamma, beta, eps=1e-4):
    # x - mean / var + eps * gamma + beta
    out = []
    for row in X:
        num_elements = len(row)
        mean = sum(row) / num_elements
        var = sum((x - mean)**2 for x in row) / num_elements
        denom = math.sqrt(var + eps)
        out.append(
            [[(x - mean)/denom * gamma + b] for x, g, b in zip(row, gamma, beta)]
        )

    return out

def split_head(X, d_model, head_index):
    low, high = d_model * head_index, d_model * (head_index + 1)
    return [row[low:high] for row in X]

def scaled_dot_product_attention(Q, K, V):
    S, d_head = len(Q), len(Q[0])
    attn_scores = matmul(Q, transpose(K))
    scale = math.sqrt(d_head)
    attn_scores = [[x / scale for x in row] for row in attn_scores]

    attn_probs = []
    for i, row in enumerate(attn_scores):
        masked = [row[j] if j <= i else float("-inf") for j in range(S)]
        attn_probs.append(softmax(masked))
        
    return matmul(attn_probs, V)

def mha(d_model, d_head, n_heads, X, params):
    Q = linear(X, params["Wq"], params["bq"])
    K = linear(X, params["Wk"], params["bk"])
    V = linear(X, params["Wv"], params["bv"])
    
    attn = [
        scaled_dot_product_attention(split_head(Q, d_model, h_i),
                                     split_head(K, d_model, h_i),
                                     split_head(V, d_model, h_i))
        for h_i in range(n_heads)
    ]
    
    # we have an array where each element is the unique head,
    # we want to combine back in to d_model 
    concat = []




