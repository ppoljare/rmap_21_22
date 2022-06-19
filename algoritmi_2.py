import numpy as np
import numpy.linalg as npla
import scipy as sc
import scipy.io as sci
import scipy.linalg as scla
import matplotlib.pyplot as plt

def tf_fun_eval(A,B,C,D,s):
	n = np.size(A,0)
	x = C @ np.linalg.solve(s*np.eye(n)-A, B) + D
	return x
#end def

# (R + r_kk @ I) @ x = b
def tri_solve(R, r_kk, b):
	n = R.shape[0]
	# init x (y_k)
	x = np.zeros(n, dtype=complex)
	# U := R + r_kk @ I
	U = R + r_kk * np.eye(n)
	# for i=n..1
	for i in range(n-1, -1, -1):
		suma = 0
		# \sum_{j=i+1}^{n} (u_ij * x_j)
		for j in range(i+1, n):
			suma += U[i, j] * x[j]
		#end for j
		x[i] = (b[i] - suma) / U[i, i]
	#end for i
	return x
#end def
'''
# A.T @ X + X @ A = C
def bartels_stewart(A, C):
	n = A.shape[0]
	
	# 1. izračunaj Schurovu dekomp. matrice A.T
	## A.T = U @ R @ U*
	## ==> R = U* @ A.T @ U
	## ==> Y := U* @ X @ U
	R, U = scla.schur(A.T, output='complex')
	## C_hat := U* @ C @ U
	C_hat = U.conj().T @ C @ U
	
	# 2. riješi Y @ R* + R @ Y = C_hat
	Y = np.zeros((n,n), dtype=complex)
	## for k=n..0
	for k in range(n-1, -1, -1):
		suma = 0
		# \sum_{j=k+1}^{n} (r_kj* @ Y_j)
		for j in range(k+1, n):
			suma += R[k, j].conj() * Y[:, j]
		#end for j
		# C_hat_k - suma
		rhs = C_hat[:, k] - suma
		# riješi (R + r_kk* @ I) @ y_k = rhs
		Y[:, k] = tri_solve(R, R[k,k].conj(), rhs)
		#end for k
	#end for k
	
	# 3. vrati supstituciju Y = U* @ X @ U
	## ==> X = U @ Y @ U*
	X = U @ Y @ U.conj()
	
	return X
#end def
'''
def bartels_stewart( A, C ):
    # Rješava X*A + A.T*X + C = 0.
    
    # A je n x n matrica -> X je n x n matrica.
    n = np.size( A, 1 );
    
    X = np.zeros( ( n, n ), dtype = complex );
    
    # 1. Konverzija u Schurovu formu matrica A i B.
    # (Radi jednostavnosti, koristimo kompleksnu formu, trebalo bi realnu.)
    [T_A, Q_A] = scla.schur( A.T, output = 'complex'); # A.T = Q_A * R * Q_A.H    
    
    CC = Q_A.conj().T @ C @ Q_A;
    
    # 2. Sada rješavamo Y * T_A.T + T_A * Y + CC = 0, stupac po stupac od Y.
    for k in range( n-1,-1,-1):
        # Rješavamo trokutasti sustav:
        # (R + r_kk I)y_k = -(c_k + r_k1*x1 + r_k2*x2 + ... + r_(k-1)k*x_(k-1)).
        
        # Pripremi desnu stranu trokutastog sustava.
        rhs = -CC[:, k];
        
        for i in range(k+1, n):
            rhs = rhs - (T_A[k, i]).conj() * X[:, i];
            
        # Riješi trokutasti sustav (R + r_kk I) x_k = rhs.
        xx = tri_solve( T_A, (T_A[k, k]).conj(), rhs );
         
        X[:, k] = xx.reshape( -1 ); # Reshapeaj xx u vektor (-1 = pogodi jedinu preostalu dimenziju) 
    # Primijeni nazad ortogonalne transformacije na X.
    
    X = Q_A @ X @ Q_A.conj().T;
    return X;
#end def

def norm_H2(A, B, C):
	P = bartels_stewart(A.T, B @ B.T)
	return np.sqrt(np.abs(np.trace(C @ P @ C.T)))
#end def

def get_gammas(A, B, C, D):
	P = bartels_stewart(A.T, B @ B.T)
	Q = bartels_stewart(A, C.T @ C)
	
	# array, matrix
	eigs, _ = npla.eig(P @ Q)
	sigma_H = np.sqrt(eigs)
	# matrix, array, matrix
	_, Sigma_D, npla.svd(D)
	
	gamma_L = max(Sigma_D[0], sigma_H[0])
	gamma_U = Sigma_D[0] + 2 * np.sum(sigma_H)
	return np.abs(gamma_L), np.abs(gamma_U)
#end def

def norm_Hinf(A, B, C, D, eps=5e-10):
	n = A.shape[0]
	gamma_L, gamma_U = get_gammas(A, B, C, D)
	gamma = (gamma_U + gamma_L) / 2
	while (gamma_U - gamma_L) > (2 * eps * gamma_L):
		gamma = (gamma_U + gamma_L) / 2
		R_inv = npla.inv(gamma**2 - D.T @ D)
		M_gamma = np.block([
			[
				A + B @ R_inv @ D.T @ C,
				B @ R_inv @ B.T
			],
			[
				-C.T @ (np.eye(D.shape[1]) + D @ R_inv @ D.T) @ C,
				-(A + B @ R_inv @ D.T @ C).T
			]
		])
		# array, matrix
		eigs, _ = npla.eig(M_gamma)
		if np.min(np.abs(np.real(eigs))) < 1e-8:
			gamma_L = gamma
		else:
			gamma_U = gamma
		#end if
	#end while
	return gamma
#end def

def modal_truncation_upper_bound(c, b, eigs):
	n = b.shape[0]
	suma = 0
	for i in range(n):
		suma += npla.norm(c[:,i] * npla.norm(b[i,:])) / np.abs(np.real(eigs[i,i]))
	#end for i
	return suma
#end def

def modal_truncation(A, B, C, D, r):
	# array, matrix
	eigs, T = npla.eig(A)
	Tinv = npla.inv(T)
	
	A_hat = Tinv @ A @ T
	B_hat = Tinv @ B
	C_hat = C @ T
	
	idx = sorted(range(len(eigs)), key=lambda k: np.abs(eigs[k]))
	
	A_hat = A_hat[idx,:][:,idx]
	B_hat = B_hat[idx,:]
	C_hat = C_hat[:,idx]
	
	A_r = A_hat[:r, :r]
	A_2 = A_hat[r:, r:]
	
	B_r = A_hat[:r, :]
	B_2 = A_hat[r:, :]
	
	C_r = A_hat[:, :r]
	C_2 = A_hat[:, r:]
	
	upper_bound_error = modal_truncation_upper_bound(C_2, B_2, A_2)
	real_error = norm_Hinf(A_2, B_2, C_2, D-D)
	
	return A_r, B_r, C_r, upper_bound_error, real_error
#end def

def balanced_truncation(A, B, C, D, r):
	# 1. riješiti Ljapunovljeve jednadžbe
	P = bartels_stewart(A.T, B @ B.T)
	Q = bartels_stewart(A, C.T @ C)
	
	# 2. Cholesky dekompozicija
	R = npla.cholesky(P)
	L = npla.cholesky(Q)
	
	# 3. SVD:  L.T @ R = U @ Sigma @ V*
	## matrix, array, matrix
	U, Sigma, V = npla.svd(L.T @ R)
	Sigma = np.diag(Sigma)
	V = V.T.conj()
	
	# 4.
	#    T = R @ V @ inv(sqrt(Sigma))
	# Tinv = inv(sqrt(Sigma)) @ U.T @ L.T
	Sigma_sqrt_inv = npla.inv(np.sqrt(Sigma))
	T = R @ V @ Sigma_sqrt_inv
	Tinv = Sigma_sqrt_inv @ U.T @ L.T
	
	# 5. zapišemo balansiranu transformaciju
	A_hat = Tinv @ A @ T
	B_hat = Tinv @ B
	C_hat = C @ T
	
	A_r = A_hat[:r, :r]
	B_r = B_hat[:r, :]
	C_r = C_hat[:, :r]
	
	# gornja ocjena pogreške:  2 * Sigma[r+1]
	## ("prva zaboravljena")
	upper_bound_error = 2 * Sigma[r]
	return A_r, B_r, C_r, upper_bound_error
#end def

def Smith_iter(A, W, p, max_iter, tol):
	n = A.shape[0]
	X = np.zeros((n,n), dtype=complex)
	i = 0
	res = npla.norm(A@X + X@A.T + W)
	while i < max_iter and res > tol:
		# inv(A + pI)
		App_inv = npla.inv(A + p*np.eye(n))
		# C(p)
		Cp = (A - np.conj(p)*np.eye(n)) @ App_inv
		# W(p)
		Wp = -2 * np.real(p) * App_inv @ W @ App_inv.conj().T
		# X = C(p) @ X @ C(p)* + W(p)
		X = Cp @ X @ Cp.conj().T + Wp
		# update state
		i += 1
		res = npla.norm(A@X + X@A.T + W)
	#end while
	if i >= max_iter:
		print('Dostigli smo maksimalan broj iteracija.')
	if res <= tol:
		print('Konvergirali smo u', i, 'koraka.')
		print('Norma reziduala:', res)
	#end if
	return X, res
#end def

def ADI_iter(A, W, p, max_iter, tol):
	n = A.shape[0]
	X = np.zeros((n,n), dtype=complex)
	i = 0
	res = npla.norm(A@X + X@A.T + W)
	while i < max_iter and res > tol:
		k = i % len(p)
		# inv(A + pI)
		App_inv = npla.inv(A + p*np.eye(n))
		# C(p)
		Cp = (A - np.conj(p[k])*np.eye(n)) @ App_inv
		# W(p)
		Wp = -2 * np.real(p[k]) * App_inv @ W @ App_inv.conj().T
		# X = C(p) @ X @ C(p)* + W(p)
		X = Cp @ X @ Cp.conj().T + Wp
		# update state
		i += 1
		res = npla.norm(A@X + X@A.T + W)
	#end while
	if i >= max_iter:
		print('Dostigli smo maksimalan broj iteracija.')
	if res <= tol:
		print('Konvergirali smo u', i, 'koraka.')
		print('Norma reziduala:', res)
	#end if
	return X, res
#end def

def LRCF_ADI(A, B, p, max_iter=3000, tol=1e-8):
	n = A.shape[0]
	
	# 1. j=1, W_0 = B, Z_0 = []
	j = 0
	W = np.copy(B)
	Z = np.empty(shape=(n,0))
	
	# 2. while norm(W_j) >= tol
	while npla.norm(W) >= tol and j < max_iter:
		k = j % len(p)
		
		# 3. V_j = inv(A + p_k*I) @ W_{j-1}
		V = npla.inv(A + p[k]*np.eye(n)) @ W
		
		# 4. W_j = W_{j-1} - 2 * Re(p_k) @ V_j
		W = W - 2 * np.real(p[k]) * V
		
		# 5. Z_j = [Z_{j-1}, sqrt(-2*Re(p_k)) * V_j]
		Z = np.block([Z, np.sqrt(-2*np.real(p[k]))*V])
		
		# 6. j = j+1
		j += 1
	#end while
	
	print(npla.norm(W))
	
	# A@X + X@A.T ~=~ -B @ B.T
	# X ~=~ Z @ Z*
	res = npla.norm(A @ (Z @ Z.conj().T) + (Z @ Z.conj().T) @ A.T + B @ B.T)
	
	return Z, res
#end def

def balanced_truncation_ADI(A, B, C, D, r_max, p):
	# 1. korištenjem LRCF_ADI dobijemo P=R@R*, Q=L@L*
	R, _ = LRCF_ADI(A, B, p)
	L, _ = LRCF_ADI(A.T, C.T, p)
	
	# 2. r := min {r_R, r_L, r_max}
	r_R = np.shape(R)[0]
	r_L = np.shape(L)[0]
	r = min(r_L, r_R, r_max)
	
	# 3. izračunaj SVD od L[:, :r].T @ R[:, :r]
	# matrix, array, matrix
	U, Sigma, V = npla.svd(L[:, :r].T @ R[:, :r])
	Sigma = np.diag(Sigma)
	V = V.T.conj()
	
	# 4.
	## T = R[:, :r] @ V @ inv(sqrt(Sigma))
	## W = L[:, :r] @ U @ inv(sqrt(Sigma))
	Sigma_sqrt_inv = npla.inv(np.sqrt(Sigma))
	T = R[:, :r] @ V @ Sigma_sqrt_inv
	W = L[:, :r] @ U @ Sigma_sqrt_inv
	
	# 5. izračunamo transformacije
	A_r = W.T @ A @ T
	B_r = W.T @ B
	C_r = C @ T
	
	# ovaj algoritam nema ocjenu pogreške
	return A_r, B_r, C_r
#end def

#### test LR_CF_ADI

n = 100
a = -100
b = -0.01 
A = np.random.rand(n,n)
Q, _ = npla.qr(A)
A = Q @ np.diag(np.linspace(a,b,n)) @ np.conj(Q.T)


B = np.eye(n, 2)
C = np.eye(n, 2)
D = np.zeros((n,n))

pADI = np.array([-87.4374, -39.1750, -14.0718, -4.8983, -1.6985, 
                  -0.5887, -0.2042, -0.0711, -0.0255, -0.0114])



Z, res = LRCF_ADI(A, B, pADI, 40, 1e-8)




## iss primjer
mat = sci.loadmat('iss.mat')


A = mat['A']
B = mat['B']
C = mat['C']
D = mat['D']

A = sc.sparse.csr_matrix.toarray(A)
B = sc.sparse.csr_matrix.toarray(B)
C = sc.sparse.csr_matrix.toarray(C)
D = sc.sparse.csr_matrix.toarray(D)

n = np.size(A, 0)
m = np.size(B, 1)

pADI = np.array([-0.6235, -3.9142, -5.8632, -13.3691, -9.7702, -30.5680, 
                  -2.4908, -56.0940, -42.2164, -33.8856, -42.9668, -58.5600])



t2 = np.logspace( -1, 3, 1000 );
tfplot_iss = np.array([abs(tf_fun_eval(A, B, C, D, tt*1j)) for tt in t2])


print("balansirano rezanje")
Ar_balanced, Br_balanced, Cr_balanced = balanced_truncation_ADI(A, B, C, D, 100, pADI)
tfplot_r_balanced_issADI = np.array([abs(tf_fun_eval(Ar_balanced, Br_balanced, Cr_balanced, D, tt*1j)) for tt in t2])

Ar_balanced, Br_balanced, Cr_balanced, bound_balanced = balanced_truncation(A, B, C, D, 20)
tfplot_r_balanced_iss = np.array([abs(tf_fun_eval(Ar_balanced, Br_balanced, Cr_balanced, D, tt*1j)) for tt in t2])


f2 = plt.figure();
plt.semilogx( t2, 20*np.log10(tfplot_iss[:,0,1]), 'r--', label='tf za input 1' );
plt.semilogx( t2, 20*np.log10(tfplot_r_balanced_issADI[:,0,1]), 'b--', label='tf_r_balanced_ADI za input 1' );
plt.semilogx( t2, 20*np.log10(tfplot_r_balanced_iss[:,0,1]), 'g--', label='tf_r_balanced za input 1' );

# CD player 

mat = sci.loadmat('CDplayer.mat')

A = mat['A']
B = mat['B']
C = mat['C']
D = mat['D']

A = sc.sparse.csr_matrix.toarray(A)

pADI = np.array([-2534.6235, -1503.9142, -586.32, -133.691, -977.02, -3056.80, 
                  -249.08, -560.940, -4221.64, -3388.56, -42.9668, -58.5600])


'''sysCD = con.ss(A, B, C, D)
tf_CD = con.ss2tf(A, B, C, D)'''
t2 = np.logspace( -1, 6, 10000 );
tfplot_CD = np.array([abs(tf_fun_eval(A, B, C, D, tt*1j)) for tt in t2])

Ar_balanced, Br_balanced, Cr_balanced = balanced_truncation_ADI(A, B, C, D, 100, pADI)
tfplot_r_balanced_CD_ADI = np.array([abs(tf_fun_eval(Ar_balanced, Br_balanced, Cr_balanced, D, tt*1j)) for tt in t2])

Ar_balanced, Br_balanced, Cr_balanced, bound_balanced = balanced_truncation(A, B, C, D, 30)
tfplot_r_balanced_CD = np.array([abs(tf_fun_eval(Ar_balanced, Br_balanced, Cr_balanced, D, tt*1j)) for tt in t2])


f2 = plt.figure();
plt.semilogx( t2, 20*np.log10(tfplot_CD[:,0,1]), 'r--', label='tf za input 1' );
plt.semilogx( t2, 20*np.log10(tfplot_r_balanced_CD_ADI[:,0,1]), 'b--', label='tf_r_balanced_ADI za input 1' );
plt.semilogx( t2, 20*np.log10(tfplot_r_balanced_CD[:,0,1]), 'g--', label='tf_r_balanced za input 1' );

plt.show()