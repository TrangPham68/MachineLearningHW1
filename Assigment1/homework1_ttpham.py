import numpy as np


def problem1 (A, B):
    '''
    Given matrices A and B, compute and return an expression for A + B.
    '''
    return A + B # A+B

def problem2 (A, B, C):
    '''
    Given matrices A, B, and C, compute and return AB 􀀀 C (i.e., right-multiply matrix A by matrix
    B, and then subtract C). Use dot or np.dot.
    '''
    return np.dot(A, B) - C # AB + transpose(C)

def problem3 (A, B, C):
    '''
    Given matrices A, B, and C, return A  B + C>, where  represents the element-wise (Hadamard)
    product and > represents matrix transpose. In numpy, the element-wise product is obtained simply
    with *
    '''
    return A*B + C.T

def problem4 (x, S, y):
    '''
    Given column vectors x and y and square matrix S, compute x^T Sy.
    '''
    return np.dot(np.dot(x.T, S), y)

def problem5 (A):
    '''
    Given matrix A, return a vector with the same number of rows as A but that contains all ones. Use np.ones.
    '''
    return np.ones(A.shape)

def problem6 (A):
    '''
    Given matrix A, return a matrix with the same shape and contents as A except that the diagonal
    terms (Aii for every valid i) are all zero.
    '''
    m = np.array(A)
    np.fill_diagonal(m, 0)
    return m

def problem7 (A, alpha):
    '''
    Given square matrix A and (scalar) , compute A+ I, where I is the identity matrix with the same
    dimensions as A. Use np.eye.
    '''
    return A + alpha * np.eye(A.shape[0])

def problem8 (A, j, i):
    '''
    Given matrix A and integers i; j, return the ith column of the jth row of A, i.e., Aji
    '''
    return A[i][j]  #i column and j row

def problem9 (A, i):
    '''
    Given matrix A and integer i, return the sum of all the entries in the ith row, i.e., Do not
    use a loop, which in Python is very slow. Instead use the np.sum function. [
    '''
    return np.sum(A[i]) #sum of everything in ith row

def problem10 (A, c, d):
    '''
    Given matrix A and scalars c; d, compute the arithmetic mean (you can use np.mean) over all entries
    of A that are between c and d (inclusive). In other words, if S = f(i; j) : c  Aij  dg, then compute
    '''
    greater = A[np.nonzero(A>=c)]
    less = greater[np.nonzero(greater<=d)]
    return np.mean(less)

def problem11 (A, k):
    '''
    Given an (n  n) matrix A and integer k, return an (n  k) matrix containing the right-eigenvectors
    of A corresponding to the k eigenvalues of A with the largest magnitude. Use np.linalg.eig to
    compute eigenvectors. [
    '''
    eigVector = np.linalg.eig(A)[1]
    col = A.shape[0] - k
    return eigVector[:, col:]

def problem12 (A, x):
    '''
    Given square matrix A and column vector x, use np.linalg.solve to compute A􀀀1x. Do not use
    np.linalg.inv or ** -1 to compute the inverse explicitly; this is numerically unstable and can, in
    some situations, give incorrect results. [
    '''
    return np.linalg.solve(A,x)

def problem13 (x, k):
    '''
    Given an n-vector x and a non-negative integer k, return a col(x) by k matrix consisting of k copies of x.
    You can use numpy methods such as np.newaxis, np.atleast 2d, and/or np.repeat.
    '''
    return x[None, :] * np.ones(k,)[:, None]

def problem14 (A):
    '''
    Given a matrix A with n rows, return a matrix that results from randomly permuting (use
    np.random.permutation) the rows (but not the columns) in A. Do not modify the input array
    A.
    '''
    m= np.array(A)
    return np.apply_along_axis(np.random.permutation, 0 , m)

if __name__ == "__main__":
    A = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
    B = A
    C = np.array([[1,1,1],
                 [1,1,1],
                 [1,1,1]])

    x = np.array([1,2,3]).T
    y = x
    alpha = 4
    i = 0
    j = 2
    c = 3
    d = 8
    k = 2

    print (A)
    print (B)
    print (C)
    print (x)

    print("Q1")
    print(problem1(A,B))
    print("Q2")
    print(problem2(A,B,C))
    print("Q3")
    print(problem3(A,B,C))
    print("Q4")
    print(problem4(x,A,y))
    print("Q5")
    print(problem5(A))
    print("Q6")
    print(problem6(A))
    print("Q7")
    print(problem7(A, alpha))
    print("Q8")
    print(problem8(A, i,j))
    print("Q9")
    print(problem9(A, i))
    print("Q10")
    print(problem10(A,c,d))
    print("Q11")
    print(problem11(A, k))
    print("Q12")
    print(problem12(A, x))
    print("Q13")
    print(problem13(x, k))
    print("Q14")
    print(problem14(A))



