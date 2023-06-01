import torch
import numpy as np
from scipy.special import erf,erfc,gamma
import torch.distributions as td

def cayley(u,v):
    x = (u**2 + v**2-1)/(u**2+(1+v)**2)
    y = -2*u/(u**2+(1+v)**2)   
    return (x,y)

def clip_by_norm(x, clip_norm, dim):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    scale = torch.clamp(clip_norm / (norm + 1e-5), min=0, max=1)
    return x * scale

def project_hyp_vec(x, c):
    return clip_by_norm(x, clip_norm=(1. - 1e-5) / np.sqrt(c), dim=1)

def np_project_hyp_vec(x, c):
    clip_norm = (1.-1e-5)/np.sqrt(c)
    x_norm = np.linalg.norm(x,axis=1)
    return x * clip_norm/ np.expand_dims(np.maximum(x_norm,clip_norm),axis=1)

# Real x, not vector!
def tf_atanh(x):
    return torch.atanh(torch.clamp(x, max=1. - 1e-15)) # Only works for positive real x.

def tf_dot(x, y):
    dot = torch.sum(x * y, dim=1, keepdim=True)
    return dot

def tf_norm(x):
    norm = torch.norm(x, 2, dim = 1, keepdim=True)
    return norm

# Real x, not vector!
def tf_tanh(x):
    return torch.tanh(torch.clamp(x, -15.0, 15.0))

def riemannian_gradient_c(u, c):
    return torch.squeeze((1-c*tf_norm(u.unsqueeze(0))**2)**2/4.0,0)

def tf_lambda_x(x, c):
    return 2. / (1 - c * tf_dot(x,x))

def mobius_scalar_mul(r, v, c=1.0):
    v = v + 1e-7
    norm_v = torch.norm(v,2,dim=1,keepdim=True)
    nomin = tf_tanh(r * tf_atanh(np.sqrt(c) * norm_v))
    result= nomin / (np.sqrt(c) * norm_v) * v
    return project_hyp_vec(result, c)

def mobius_add(u, v, c):
    v = v + 1e-7
    tf_dot_u_v = 2. * c * tf_dot(u, v)
    tf_norm_u_sq = c * tf_dot(u,u)
    tf_norm_v_sq = c * tf_dot(v,v)
    denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
    result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
    return project_hyp_vec(result, c)

def mobius_lin(M,x,c=1.0):
    #Mx = tf.matmul(x,M)
    Mx = torch.einsum('ijk,ik->ik',M,x)       
    Mx_norm = torch.norm(Mx,2,dim=1,keepdim=True)
    x_norm = torch.norm(x,2,dim=1,keepdim=True)
    t = 1/np.sqrt(c)*tf_tanh(Mx_norm/x_norm*tf_atanh(np.sqrt(c)*x_norm))*Mx/Mx_norm
    return project_hyp_vec(t,c)

def np_dist_hp(x,y):
    d_xy = np.linalg.norm(x-y,2,axis=1)
    return np.arccosh(1+d_xy**2/(2*x[:,-1]*y[:,-1]))

def np_dist_pd(x,y):
    x_norm = np.linalg.norm(x,2,axis=1)
    y_norm = np.linalg.norm(y,2,axis=1)
    d_xy = np.linalg.norm(x-y,2,axis=1)
    return np.arccosh(1+2*(d_xy**2)/((1-x_norm**2)*(1-y_norm**2)))

def dist_hp(x,y):
    d_xy = torch.norm(x-y,2,dim=1)
    return torch.acosh(1+d_xy**2/(2*x[:,-1]*y[:,-1]))

def dist_pd(x,y):
    x_norm = torch.norm(x,2,dim=1)
    y_norm = torch.norm(y,2,dim=1)
    d_xy = torch.norm(x-y,2,dim=1)
    return torch.acosh(1+2*d_xy**2/((1-x_norm**2)*(1-y_norm**2)))

def dist_pd_c(x,y,c=1.0):
    return 2/np.sqrt(c)*tf_atanh(np.sqrt(c)*torch.norm(mobius_add(-x,y,c),2,dim=1))

def exp_map(v,c=1.0):
    v = v + 1e-7 # against nan
    norm_v = tf_norm(v)
    res = tf_tanh(np.sqrt(c)*norm_v)*v/(np.sqrt(c)*norm_v)
    return project_hyp_vec(res,c)

def lambda_c(x,c):
    return 2/(1-torch.norm(x,2,dim=1)**2)

def exp_map_x_(x,v,c=1.0):
    v = v + 1e-7
    norm_v = tf_norm(v)
    return mobius_add(x,tf_tanh(np.sqrt(c)*lambda_c(x,c)*norm_v)*v/(np.sqrt(c)*norm_v),c)

def exp_map_x(x, v, c):
    v = v + 1e-7 # against nan
    norm_v = tf_norm(v)
    second_term = (tf_tanh(np.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return mobius_add(x, second_term, c)

def np_exp_map(v,c=1.0):
    v = v + 1e-7 # against nan
    res = np.tanh(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))*v/(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))
    return np_project_hyp_vec(res,c)

def np_log_map(v,c=1.0):
    #v = v + 1e-7 # against nan
    norm_v = np.linalg.norm(v,2,axis=1,keepdims=True)
    return np.arctanh(np.sqrt(c)*norm_v)*v/(np.sqrt(c)*norm_v)

def log_map(v,c=1.0):
    norm_v = torch.norm(v,2,dim=1,keepdim=True)
    res = tf_atanh(np.sqrt(c)*norm_v)*v/(np.sqrt(c)*norm_v)
    return res

def log_map_x(x, y, c):
    diff = mob_add(-x, y, c)
    lam = lambda_x(x, c)
    return 2. / (np.sqrt(c) * lam) * np.arctanh(np.sqrt(c) * np.linalg.norm(diff)) / (np.linalg.norm(diff)) * diff

def tf_log_map_x(x, y, c):
    diff = mobius_add(-x, y, c) + 1e-7
    norm_diff = tf_norm(diff)
    lam = tf_lambda_x(x, c)
    return ( ( (2. / np.sqrt(c)) / lam) * tf_atanh(np.sqrt(c) * norm_diff) / norm_diff) * diff

def np_gauss_prob_halfplane(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-np_dist_hp(x,mu)**2/(2*sigma**2))
    return p

def np_gauss_prob_pd(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-np_dist_pd(x,mu**2)/(2*sigma**2))
    return p

def np_gauss_prob_pd_r(r,sigma):
    # using poincaré half-plane
    Z = np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-r/(2*sigma**2))*np.sinh(r)
    return p

def gauss_prob_halfplane(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*torch.exp(sigma**2/2)*torch.erf(sigma/np.sqrt(2.0))
    p = 1/Z*torch.exp(-dist_hp(x,mu)**2/(2*sigma**2))
    return p

def gauss_prob_pd(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*torch.exp(sigma**2/2)*torch.erf(sigma/np.sqrt(2.0))
    p = 1/Z*torch.exp(-dist_pd(x,mu)**2/(2*sigma**2))
    return p

def gauss_prob_pd_r(r,sigma):
    # using poincaré half-plane
    sigma_t = torch.Tensor([sigma])
    Z = np.sqrt(np.pi/2)*sigma*torch.exp(sigma_t**2/2)*torch.erf(sigma_t/np.sqrt(2.0))
    p = 1/Z*torch.exp(-r**2/(2*sigma**2))*torch.sinh(r)
    return p

def np_gauss_prob_pd_r(r,sigma):
    # using poincaré half-plane
    Z = np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-r**2/(2*sigma**2))*np.sinh(r)
    return p

def np_sample_gauss_pd(shape,oversample=5,radius=0.85,sigma=1.0):
    # rejection sample in Poincaré half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    #unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    unif_samples = np.stack([r*np.cos(phi), r*np.sin(phi)],axis=1)

    # zero mean, unit sigma in disk
    mean = np.array([[0.0,0.0]])
    p_samples = np_gauss_prob_pd(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = np.max(p_samples)
    accepted = np.random.rand(n_samples) < (p_samples/max_value)
    # select the samples - make sure it's enough
    u = unif_samples[:,0][accepted]
    v = unif_samples[:,1][accepted]
    # transform samples using cayley mapping z = (i-w)/(i+w)
    disk_samples = np.stack([u,v],axis=1)
    idx = (np.prod(shape)/2).astype(np.int32)
    return np.reshape(disk_samples[:idx],shape)

def np_sample_gauss_hn(shape,oversample=5,support_radius=0.85,sigma=1.0):
    # rejection sample in Poincaré disk
    n = shape[0]
    d = shape[1]
    n_samples = n*oversample
    direction = np.random.randn(n_samples,d)
    direction = direction/np.linalg.norm(direction,axis=1,keepdims=True)
    p = np.random.rand(n_samples)
    r = (p*support_radius**d)**(1/d)
    f_r = np_gauss_prob_pd_r(r,sigma)
    g_r = np.exp(r) # proposal
    M = np.max(f_r/g_r) # empirical M
    accepted = np.random.rand(n_samples) < (f_r/(M*g_r))
    r_a = r[accepted]
    dir_a = direction[accepted]

    samples = np.expand_dims(r_a,1)*dir_a
    unif_samples = np.expand_dims(r,1)*direction
    return np.reshape(samples[:n],shape), np.reshape(unif_samples[:n],shape)

def np_sample_uniform_h(n_samples,radius=1.0,sigma=1.0):
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1))
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    return unif_samples

def np_sample_uniform_h_r(n_samples,radius=1.0,sigma=1.0):
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1))
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    return unif_samples

def np_sample_unit_sphere(n,d):
    x = np.random.randn(n,d)
    sphere = x/np.linalg.norm(x,axis=1,keepdims=True)
    return sphere

def np_sample_uniform_h_cosh(n_samples,radius=1.0,sigma=1.0):
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1))
    #unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
    unif_samples = np.stack([np.cosh(r)*np.cos(phi), np.cosh(r)*np.sin(phi)],axis=1)
    return unif_samples

def sample_unit_sphere(n,d):
    x = torch.randn([n,d])
    sphere = x/torch.norm(x,dim=1,keepdim=True)
    return sphere
   
def sample_gauss_pd(shape,oversample=5,radius=0.85,sigma=1.0):
    # rejection sample in Poincaré disk
    d = shape[1]
    n_samples = torch.prod(shape)*oversample
    phi = torch.rand([n_samples])*np.pi
    p = torch.rand([n_samples])
    r = torch.acosh(1+p*(torch.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = torch.stack([torch.sinh(r)*torch.cos(phi), torch.sinh(r)*torch.sin(phi)],dim=1)
    #unif_samples = tf.stack([r*tf.cos(phi), r*tf.sin(phi)],axis=1)

    # zero mean, unit sigma in half plane
    mean = torch.FloatTensor([[0.0,0.0]])
    
    p_samples = gauss_prob_pd(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = torch.max(p_samples)
    accepted = torch.squeeze(torch.where(torch.rand([n_samples]) < (p_samples/max_value)),1)
    #accepted = tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value),unif_samples,tf.zeros_like(unif_samples))
    # select the samples - make sure it's enough
    u = torch.masked_select(unif_samples[:,0],accepted)
    v = torch.masked_select(unif_samples[:,1],accepted)
    # transform samples using cayley mapping z = (w-i)/(w+i) - project to unit disk
    disk_samples = torch.stack([u,v],dim=1)
    idx = torch.range(torch.prod(shape)/2)
    return torch.reshape(torch.gather(disk_samples,idx),shape)

def sample_gauss_hn_(shape,oversample=5,radius=1.0,sigma=1.0):
    # rejection sample in Poincaré disk
    n = shape[0] # batch_sizes
    d = shape[1]    
    n_samples = shape[0]*oversample
    p = torch.rand([n_samples])
    #r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    r = torch.pow((p*np.power(radius,d)),(1/d))
    f_r = gauss_prob_pd_r(r,sigma)
    g_r = 1/radius#tf.exp(r) # proposal
    
    M = torch.max(f_r/g_r) # empirical M
    accepted = torch.rand([n_samples]) < (f_r/(M*g_r))
    r_a = torch.masked_select(r,accepted)
    #dir_a = tf.boolean_mask(direction,accepted)

    direction = sample_unit_sphere(n,d)
    samples = r_a[:n].unsqueeze(1)*direction
    unif_samples = r[:n].unsqueeze(1)*direction
    return torch.reshape(samples[:n],shape)

def np_sample_gauss_hn_(shape,oversample=5,radius=1.0,sigma=1.0):
    # rejection sample in Poincaré disk
    n = shape[0] # batch_sizes
    d = shape[1]    
    n_samples = shape[0]*oversample
    p = np.random.rand(n_samples)
    #r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    d_fl = np.cast(d,np.float32)
    r = np.pow((p*np.pow(radius,d_fl)),(1/d_fl))
    f_r = np_gauss_prob_pd_r(r,sigma)
    g_r = 1/radius#tf.exp(r) # proposal
    
    M = torch.max(f_r/g_r) # empirical M
    accepted = torch.rand([n_samples]) < (f_r/(M*g_r))
    r_a = torch.masked_select(r,accepted)
    #dir_a = tf.boolean_mask(direction,accepted)

    direction = sample_unit_sphere(n,d)
    samples = r_a[:n].unsqueeze(1)*direction
    unif_samples = r[:n].unsqueeze(1)*direction
    return torch.reshape(samples[:n],shape)

def reparam_h(mu,log_sigma,radius=0.88,c=1.0,scalar_dispersion=False,use_expmap=False,prior_var=1.0):
    sigma = torch.exp(0.5*log_sigma)
    if use_expmap:
        eps_sample = exp_map(prior_var*torch.rand(mu.shape),c)
    else:
        #prior_sample = sample_gauss_pd(tf.shape(mu),radius=radius)
        #eps_sample = project_hyp_vec(sample_gauss_pd(tf.shape(mu),radius=radius,sigma=prior_var),c)
        eps_sample = project_hyp_vec(sample_gauss_hn_(mu.shape,radius=radius,sigma=prior_var),c)
        #eps_sample = project_hyp_vec(sample_gauss_hn(tf.shape(mu),radius=radius,sigma=sigma,batch_size=batch_size),c)

    if scalar_dispersion:
        tsf_eps = mobius_scalar_mul(sigma,eps_sample,c)
    else:
        Sigma = torch.diag_embed(sigma)
        tsf_eps = mobius_lin(Sigma,eps_sample,c)
        
    z = mobius_add(mu,tsf_eps,c)
    #z = exp_map_x(mu,eps_sample,c)
    #return add_res, project_hyp_vec(prior_sample,c)
    return z, eps_sample

def reparam_h_wrapped(mu,log_sigma,c=1.0,prior_var=1.0):
    sigma = torch.exp(0.5*log_sigma)
    scaled_eps = exp_map(np.sqrt(sigma)*torch.randn(mu.shape)/tf_lambda_x(mu,c),c)
    z = exp_map_x(mu,scaled_eps,c)
    #return add_res, project_hyp_vec(prior_sample,c)
    return z, scaled_eps

def reparam_h_fullcov(mu,cov,c=1.0,prior_var=1.0):
    # TODO: try w/ Cholesky fctr.
    mvn = td.MultivariateNormal(
        loc=[0.0,0.0], covariance_matrix=cov)
    sample = mvn.sample(mu.shape[0])
    z = exp_map_x(mu,sample,c)
    return z, sample

def sq_distances(x,y,scaled=True):
    sq_norms_x = torch.sum(torch.square(x), dim=1, keepdim=True)
    sq_norms_y = torch.sum(torch.square(y), dim=1, keepdim=True)
    dotprods = torch.matmul(x, y.t())
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*||x||*||y||
    d = sq_norms_x + sq_norms_y.T - 2. * dotprods
    d_scaled = d/((1-sq_norms_x)*(1-sq_norms_y))
    d_scaled = 1+2*torch.where(d_scaled<1.0, torch.ones_like(d_scaled)+1e-10, d_scaled)
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances(x,y,q,c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances(x,y)
    #dotprods = tf.matmul(points, points, transpose_b=True)
    #sq_dist = sq_norms + tf.transpose(sq_norms) - 2. * dotprods
    hyp_dist = torch.acosh(d_xy+1e-6)**q
    return sq_norms_x, d_xy, hyp_dist

def block_diagonal(matrices, dtype=torch.float32):
    matrices = [torch.Tensor(matrix) for matrix in matrices]
    blocked_rows = 0
    blocked_cols = 0
    batch_shape = torch.Size()
    for matrix in matrices:
        full_matrix_shape = matrix.shape
        batch_shape = batch_shape + full_matrix_shape[:-2]
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = matrix.shape
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = sum(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = matrix.shape
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(torch.nn.functional.pad(
            matrix,
            (row_before_length, row_after_length, 0, 0, 0, 0),
            mode='constant',
            value=0))
    blocked = torch.cat(row_blocks, dim=-2)
    blocked_shape = batch_shape + torch.Size([blocked_rows, blocked_cols])
    blocked = blocked.reshape(blocked_shape)
    return blocked


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd