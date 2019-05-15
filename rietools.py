import tensorflow as tf
import numpy as np
from scipy.special import erf,erfc,gamma

def cayley(u,v):
    x = (u**2 + v**2-1)/(u**2+(1+v)**2)
    y = -2*u/(u**2+(1+v)**2)   
    return (x,y)

def project_hyp_vec(x, c):
    return tf.clip_by_norm(x, clip_norm=(1. - 1e-5) / tf.sqrt(c), axes=[1])

def np_project_hyp_vec(x, c):
    clip_norm = (1.-1e-5)/np.sqrt(c)
    x_norm = np.linalg.norm(x,axis=1)
    return x * clip_norm/ np.expand_dims(np.maximum(x_norm,clip_norm),axis=1)

# Real x, not vector!
def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1. - 1e-15)) # Only works for positive real x.

def tf_dot(x, y):
    dot = tf.reduce_sum(x * y, axis=1, keepdims=True)
    return dot

def tf_norm(x):
    norm = tf.norm(x, 2, axis = 1, keepdims=True)
    return norm

# Real x, not vector!
def tf_tanh(x):
    return tf.tanh(tf.minimum(tf.maximum(x, -15.0), 15.0))

def riemannian_gradient_c(u, c):
    return tf.squeeze((1-c*tf_norm(tf.expand_dims(u,0))**2)**2/4.0,0)

def tf_lambda_x(x, c):
    return 2. / (1 - c * tf_dot(x,x))

def mobius_scalar_mul(r, v, c=1.0):
    v = v + 1e-7
    norm_v = tf.norm(v,2,axis=1,keepdims=True)
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
    Mx = tf.einsum('ijk,ik->ik',M,x)       
    Mx_norm = tf.norm(Mx,2,axis=1,keepdims=True)
    x_norm = tf.norm(x,2,axis=1,keepdims=True)
    t = 1/tf.sqrt(c)*tf_tanh(Mx_norm/x_norm*tf_atanh(tf.sqrt(c)*x_norm))*Mx/Mx_norm
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
    d_xy = tf.norm(x-y,2,axis=1)
    return tf.acosh(1+d_xy**2/(2*x[:,-1]*y[:,-1]))

def dist_pd(x,y):
    x_norm = tf.norm(x,2,axis=1)
    y_norm = tf.norm(y,2,axis=1)
    d_xy = tf.norm(x-y,2,axis=1)
    return tf.acosh(1+2*d_xy**2/((1-x_norm**2)*(1-y_norm**2)))

def dist_pd_c(x,y,c=1.0):
    return 2/tf.sqrt(c)*tf_atanh(tf.sqrt(c)*tf.norm(mobius_add(-x,y,c),2,axis=1))

def exp_map(v,c=1.0):
    v = v + 1e-7 # against nan
    norm_v = tf_norm(v)
    res = tf_tanh(tf.sqrt(c)*norm_v)*v/(tf.sqrt(c)*norm_v)
    return project_hyp_vec(res,c)

def lambda_c(x,c):
    return 2/(1-tf.norm(x,2,axis=1)**2)

def exp_map_x_(x,v,c=1.0):
    v = v + 1e-7
    norm_v = tf_norm(v)
    return mobius_add(x,tf_tanh(tf.sqrt(c)*lambda_c(x,c)*norm_v)*v/(tf.sqrt(c)*norm_v),c)

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
    #norm_v = tf.Print(tf.norm(v,2,axis=1,keepdims=True),[tf.norm(v,2,axis=1,keepdims=True)])
    norm_v = tf.norm(v,2,axis=1,keepdims=True)
    return tf_atanh(tf.sqrt(c)*norm_v)*v/(tf.sqrt(c)*norm_v)

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
    Z = 2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-dist_hp(x,mu)**2/(2*sigma**2))
    return p

def gauss_prob_pd(x,mu,sigma):
    # using poincaré half-plane
    Z = 2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-dist_pd(x,mu)**2/(2*sigma**2))
    return p

def gauss_prob_pd_r(r,sigma):
    # using poincaré half-plane
    Z = tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-r**2/(2*sigma**2))*tf.sinh(r)
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
    #r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    r = (p*support_radius**d)**(1/d)
    #unif_samples = tf.stack([tf.sinh(r)*tf.cos(phi), tf.sinh(r)*tf.sin(phi)],axis=1)
    #unif_samples = tf.stack([r*tf.cos(phi), r*tf.sin(phi)],axis=1)
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
    x = tf.random_normal([n,d])
    sphere = x/tf.norm(x,axis=1,keepdims=True)
    return sphere
   
def sample_gauss_pd(shape,oversample=5,radius=0.85,sigma=1.0):
    # rejection sample in Poincaré disk
    d = shape[1]
    n_samples = tf.reduce_prod(shape)*oversample
    phi = tf.random_uniform([n_samples])*np.pi
    p = tf.random_uniform([n_samples])
    r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = tf.stack([tf.sinh(r)*tf.cos(phi), tf.sinh(r)*tf.sin(phi)],axis=1)
    #unif_samples = tf.stack([r*tf.cos(phi), r*tf.sin(phi)],axis=1)

    # zero mean, unit sigma in half plane
    mean = tf.constant(np.array([[0.0,0.0]]),tf.float32)
    
    p_samples = gauss_prob_pd(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = tf.reduce_max(p_samples)
    accepted = tf.squeeze(tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value)),1)
    #accepted = tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value),unif_samples,tf.zeros_like(unif_samples))
    # select the samples - make sure it's enough
    u = tf.boolean_mask(unif_samples[:,0],accepted)
    v = tf.boolean_mask(unif_samples[:,1],accepted)
    # transform samples using cayley mapping z = (w-i)/(w+i) - project to unit disk
    disk_samples = tf.stack([u,v],axis=1)
    idx = tf.cast(tf.range(tf.reduce_prod(shape)/2),tf.int32)
    return tf.reshape(tf.gather(disk_samples,idx),shape)

def sample_gauss_hn_(shape,oversample=5,radius=1.0,sigma=1.0):
    # rejection sample in Poincaré disk
    n = shape[0] # batch_sizes
    d = shape[1]    
    n_samples = shape[0]*oversample
    p = tf.random_uniform([n_samples])
    #r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    d_fl = tf.cast(d,tf.float32)
    r = tf.pow((p*tf.pow(radius,d_fl)),(1/d_fl))
    f_r = gauss_prob_pd_r(r,sigma)
    g_r = 1/radius#tf.exp(r) # proposal
    
    M = tf.reduce_max(f_r/g_r) # empirical M
    accepted = tf.random_uniform([n_samples]) < (f_r/(M*g_r))
    r_a = tf.boolean_mask(r,accepted)
    #dir_a = tf.boolean_mask(direction,accepted)

    direction = sample_unit_sphere(n,d)
    samples = tf.expand_dims(r_a[:n],1)*direction
    unif_samples = tf.expand_dims(r[:n],1)*direction
    return tf.reshape(samples[:n],shape)

def sample_gauss_hn__(shape,batch_size,oversample=5,radius=1.0,sigma=1.0):
    # rejection sample in Poincaré disk
    n = shape[0] # batch_sizes
    d = shape[1]    
    n_samples = shape[0]*oversample
    p = tf.random_uniform([n_samples])
    #r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    d_fl = tf.cast(d,tf.float32)
    r = tf.pow((p*tf.pow(radius,d_fl)),(1/d_fl))

    #i = tf.constant(0)
    #def body(i,outputs):
    samples_all = []
    #sigma_list = tf.unstack(sigma)
    for sig in sigma_list:
        f_r = gauss_prob_pd_r(r,sigma[i])
        g_r = 1/radius#tf.exp(r) # proposal
    
        M = tf.reduce_max(f_r/g_r) # empirical M
        accepted = tf.random_uniform([n_samples]) < (f_r/(M*g_r))
        r_a = tf.boolean_mask(r,accepted)
        samples_all.append(r_a)
        #outputs = outputs.write(i, r_a)

        #return [i, outputs]

    # do the loop:
    #outputs = tf.TensorArray(dtype=tf.float32, infer_shape=False, size=1, 
    #                 dynamic_size=True)
    #_, outputs = tf.while_loop(lambda i, *_: tf.less(i, n), body, [i, outputs])
    #outputs = outputs.stack()
    #print(outputs)
    
    r_samples = tf.stack([outputs[i][0] for i in range(len(sigma_list))])
    direction = sample_unit_sphere(n,d)
    samples = tf.expand_dims(r_samples,1)*direction 
    return samples

def reparam_h(mu,log_sigma,radius=0.88,c=1.0,scalar_dispersion=False,use_expmap=False,prior_var=1.0):
    sigma = tf.exp(0.5*log_sigma)
    if use_expmap:
        eps_sample = exp_map(prior_var*tf.random_normal(tf.shape(mu)),c)
    else:
        #prior_sample = sample_gauss_pd(tf.shape(mu),radius=radius)
        #eps_sample = project_hyp_vec(sample_gauss_pd(tf.shape(mu),radius=radius,sigma=prior_var),c)
        eps_sample = project_hyp_vec(sample_gauss_hn_(tf.shape(mu),radius=radius,sigma=prior_var),c)
        #eps_sample = project_hyp_vec(sample_gauss_hn(tf.shape(mu),radius=radius,sigma=sigma,batch_size=batch_size),c)

    if scalar_dispersion:
        tsf_eps = mobius_scalar_mul(sigma,eps_sample,c)
    else:
        Sigma = tf.matrix_diag(sigma)
        tsf_eps = mobius_lin(Sigma,eps_sample,c)
        
    z = mobius_add(mu,tsf_eps,c)
    #z = exp_map_x(mu,eps_sample,c)
    #return add_res, project_hyp_vec(prior_sample,c)
    return z, eps_sample

def reparam_h_2(mu,log_sigma,c=1.0,prior_var=1.0):
    sigma = tf.exp(0.5*log_sigma)
    scaled_eps = exp_map(sigma*tf.random_normal(tf.shape(mu))/tf_lambda_x(mu,c),c)
    z = exp_map_x(mu,scaled_eps,c)
    #return add_res, project_hyp_vec(prior_sample,c)
    return z, scaled_eps