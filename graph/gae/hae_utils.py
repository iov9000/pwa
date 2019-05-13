from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np

def cayley(u,v):
    x = (u**2 + v**2-1)/(u**2+(1+v)**2)
    y = -2*u/(u**2+(1+v)**2)   
    return (x,y)

def project_hyp_vec(x, c):
    return tf.clip_by_norm(x, clip_norm=(1. - 1e-5) / tf.sqrt(c), axes=[1])

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
    Mx = tf.einsum(u'ijk,ik->ik',M,x)       
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

#def exp_map(v,c=1.0):
#    v = v + 1e-7 # Perturbe v to avoid dealing with v = 0
#    norm_v = tf_norm(v)
#    return tf_tanh(tf.sqrt(c)*norm_v)*v/(tf.sqrt(c)*norm_v)

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
    v = v + 1e-7 # Perturbe v to avoid dealing with v = 0
    norm_v = tf_norm(v)
    second_term = (tf_tanh(np.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return mobius_add(x, second_term, c)

#def np_exp_map(v,c=1.0):
#    return np.tanh(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))*v/(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))

def np_exp_map(v,c=1.0):
    v = v + 1e-7 # against nan
    res = np.tanh(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))*v/(np.sqrt(c)*np.linalg.norm(v,2,axis=1,keepdims=True))
    return np_project_hyp_vec(res,c)


def log_map(v,c=1.0):
    #norm_v = tf.Print(tf.norm(v,2,axis=1,keepdims=True),[tf.norm(v,2,axis=1,keepdims=True)])
    norm_v = tf.norm(v,2,axis=1,keepdims=True)
    return tf_atanh(tf.sqrt(c)*norm_v)*v/(tf.sqrt(c)*norm_v)

def hyp_fflayer(x,dim,activation=None,name='',c=1.0):
    with tf.variable_scope("hyp_ff_layer_"+name):
        W = tf.get_variable('weight',shape=[x.get_shape().as_list()[1],dim],
                           initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('hypbias',shape=[dim], initializer=tf.constant_initializer(0.0))
        # input x is Euclidean -> project to disk
        x = exp_map(x,c=1.0) + 1e-7
        #b = exp_map(tf.expand_dims(b,0),c=1.0) + 1e-7
        b = tf.expand_dims(b,0)
        Mx = tf.matmul(x,W) + 1e-7
        Mx_norm = tf.norm(Mx,2,axis=1,keepdims=True)
        x_norm = tf.norm(x,2,axis=1,keepdims=True)
        y = 1./tf.sqrt(c)*tf_tanh(Mx_norm/x_norm*tf_atanh(tf.sqrt(c)*x_norm))*Mx/Mx_norm
        y = project_hyp_vec(y,c)
        y = mobius_add(y,b,c)
        if activation is not None:
            # hyperbolic non-linearity
            y = exp_map(activation(log_map(y,c)),c)
        return y

def np_gauss_prob_halfplane(x,mu,sigma):    
    # using poincare half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*np.erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-np_dist_hp(x,mu)**2/(2*sigma**2))
    return p

def np_gauss_prob_pd(x,mu,sigma):
    # using poincare half-plane
    Z = 2*np.pi*np.sqrt(np.pi/2)*sigma*np.exp(sigma**2/2)*np.erf(sigma/np.sqrt(2.0))
    p = 1/Z*np.exp(-np_dist_pd(x,mu**2)/(2*sigma**2))
    return p

def gauss_prob_halfplane(x,mu,sigma):
    # using poincare half-plane
    Z = 2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-dist_hp(x,mu)**2/(2*sigma**2))
    return p

def gauss_prob_pd(x,mu,sigma):
    # using poincare half-plane
    Z = 2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0))
    p = 1/Z*tf.exp(-dist_pd(x,mu)**2/(2*sigma**2))
    return p

def np_sample_gauss_hp(shape,oversample=5,radius=3.0,sigma=1.0):
    # rejection sample in Poincare half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)

    # zero mean, unit sigma in half plane
    mean = np.array([[0.0,1.0]])
    
    p_samples = np_gauss_prob_halfplane(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = np.max(p_samples)
    accepted = np.random.rand(n_samples) < (p_samples/max_value)
    # select the samples - make sure it's enough
    u = unif_samples[:,0][accepted]
    v = unif_samples[:,1][accepted]
    # transform samples using cayley mapping z = (i-w)/(i+w)
    x,y = cayley(u,v)    
    disk_samples = np.stack([x,y],axis=1)
    idx = (np.prod(shape)/2).astype(np.int32)
    return np.reshape(disk_samples[:idx],shape)

def np_sample_gauss_pd(shape,oversample=5,radius=0.85,sigma=1.0):
    # rejection sample in Poincare half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = np.stack([np.sinh(r)*np.cos(phi), np.sinh(r)*np.sin(phi)],axis=1)
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

def np_sample_gauss_h_2(shape,oversample=5,radius=1.0,sigma=1.0):
    # rejection sample in Poincare half-plane
    d = shape[1]
    n_samples = np.prod(shape)*oversample
    phi = np.random.rand(n_samples)*np.pi
    #p = np.random.rand(n_samples)
    #r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    x = 2*radius*np.random.rand(n_samples)-radius
    y = radius*np.random.rand(n_samples)
    unif_samples = np.stack([x, y],axis=1)

    # zero mean, unit sigma in half plane
    mean = np.array([[0.0,1.0]])
    
    p_samples = np_gauss_prob_halfplane(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = np.max(p_samples)
    accepted = np.random.rand(n_samples) < (p_samples/max_value)
    #accepted = tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value),unif_samples,tf.zeros_like(unif_samples))
    # select the samples - make sure it's enough
    u = unif_samples[:,0][accepted]
    v = unif_samples[:,1][accepted]
    # transform samples using cayley mapping z = (w-i)/(w+i)
    x,y = cayley(u,v)
    #disk_samples = tf.stack([x,y],axis=1)/tf.cast(n_samples,tf.float32)
    disk_samples = np.stack([x,y],axis=1)
    disk_samples_hp = np.stack([u,v],axis=1)
    idx = (np.prod(shape)/2).astype(np.int32)
    return np.reshape(disk_samples[:idx],shape)

def np_sample_uniform_h(shape,radius=1.0,sigma=1.0):
    # rejection sample in Poincare half-plane
    d = shape[1]
    n_samples = np.prod(shape)
    phi = np.random.rand(n_samples)*2*np.pi
    p = np.random.rand(n_samples)
    r = np.arccosh(1+p*(np.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = np.stack([r*np.cos(phi), r*np.sin(phi)],axis=1)

    return unif_samples, r

def sample_gauss_hp(shape,oversample=5,radius=1.0):
    # rejection sample in Poincare half-plane
    d = shape[1]
    n_samples = tf.reduce_prod(shape)*oversample
    phi = tf.random_uniform([n_samples])*np.pi
    p = tf.random_uniform([n_samples])
    r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = tf.stack([tf.sinh(r)*tf.cos(phi), tf.sinh(r)*tf.sin(phi)],axis=1)
    
    # zero mean, unit sigma in half plane
    mean = tf.constant(np.array([[0.0,1.0]]),tf.float32)
    sigma = 1.0
    
    p_samples = gauss_prob_halfplane(unif_samples,mean,sigma)
    # accept in proportion to highest
    max_value = tf.reduce_max(p_samples)
    accepted = tf.squeeze(tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value)),1)
    #accepted = tf.where(tf.random_uniform([n_samples]) < (p_samples/max_value),unif_samples,tf.zeros_like(unif_samples))
    # select the samples - make sure it's enough
    u = tf.boolean_mask(unif_samples[:,0],accepted)
    v = tf.boolean_mask(unif_samples[:,1],accepted)
    # transform samples using cayley mapping z = (w-i)/(w+i) - project to unit disk
    x,y = cayley(u,v)    
    #disk_samples = tf.stack([x,y],axis=1)/tf.cast(n_samples,tf.float32)
    disk_samples = tf.stack([x,y],axis=1)
    disk_samples_hp = tf.stack([u,v],axis=1)
    idx = tf.cast(tf.range(tf.reduce_prod(shape)/2),tf.int32)
    return tf.reshape(tf.gather(disk_samples,idx),shape)

def sample_gauss_pd(shape,oversample=5,radius=0.85):
    # rejection sample in Poincare disk
    d = shape[1]
    n_samples = tf.reduce_prod(shape)*oversample
    phi = tf.random_uniform([n_samples])*2*np.pi
    p = tf.random_uniform([n_samples])
    r = tf.acosh(1+p*(tf.cosh(radius-1e-5)-1)) # support (-3sigma, 3sigma)
    unif_samples = tf.stack([tf.sinh(r)*tf.cos(phi), tf.sinh(r)*tf.sin(phi)],axis=1)
    
    # zero mean, unit sigma in half plane
    mean = tf.constant(np.array([[0.0,0.0]]),tf.float32)
    sigma = 1.0
    
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

def reparam_h(mu,sigma,c=1.0,use_expmap=False):
    print('Using exponential map of Euclidean Gaussian:', use_expmap)
    if use_expmap:
        prior_sample = exp_map(tf.random_normal(tf.shape(mu)),c)
    else:
        #prior_sample = exp_map(sample_gauss_h(tf.shape(mu)))
        prior_sample = project_hyp_vec(sample_gauss_pd(tf.shape(mu),radius=0.85),c)
        #prior_sample = project_hyp_vec(sample_gauss_hp(tf.shape(mu),radius=3.0),c)
    sigma = tf.exp(0.5*sigma)
    Sigma = tf.matrix_diag(sigma)
    tsf_eps = mobius_lin(Sigma,prior_sample,c)
    add_res = mobius_add(mu,tsf_eps,c)
    #return add_res, project_hyp_vec(prior_sample,c)
    return add_res, prior_sample

def sq_distances(x,y,scaled=True):
    sq_norms_x = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    sq_norms_y = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
    dotprods = tf.matmul(x, y, transpose_b=True)
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*||x||*||y||
    d = sq_norms_x + tf.transpose(sq_norms_y) - 2. * dotprods
    d_scaled = d/((1-sq_norms_x)*(1-sq_norms_y))
    d_scaled = 1+2*tf.where(d_scaled<1.0, tf.ones_like(d_scaled)+1e-10, d_scaled)
    return sq_norms_x, sq_norms_y, d_scaled

def hyp_distances(x,y,q,c=1.0):
    sq_norms_x, sq_norms_y, d_xy = sq_distances(x,y)
    #dotprods = tf.matmul(points, points, transpose_b=True)
    #sq_dist = sq_norms + tf.transpose(sq_norms) - 2. * dotprods
    hyp_dist = tf.acosh(d_xy+1e-6)**q
    return sq_norms_x, d_xy, hyp_dist

def mmd_penalty(params, N, sample_pz, sample_qhat,use_expmap):
    # assert opts["e_noise"] in ('gaussian'), \
        # 'MMD++ works only with Gaussian encoders!'
    # Number of codes per the same pic
    ns = params[u'nr_mmd_samples']
    #n = params[u'batch_size']
    ls = params[u'embedding_dim']
    kernel = params[u'mmd_kernel']
    sigma2_p = 1 # params['pz_scale'] ** 2
    params[u'pz'] = u'normal'

    n = tf.cast(tf.shape(sample_pz)[0],tf.float32)
    #mu_tiled = tf.tile(mu,[n,1])
    #sigma_tiled = tf.tile(sigma,[n,1])
    # sample_qhat, sample_pz = reparam_h(mu_tiled,sigma_tiled,radius=3.0)
    # q=1 -> using Laplacian kernel
    #sample_qhat, sample_pz = reparam_h(mu,sigma,radius=0.85,use_expmap=use_expmap)
    sq_norms_pz, _, dist_pz = hyp_distances(sample_pz,sample_pz,1,params[u'c'])
    sq_norms_qhat, dist_qhat_prime, dist_qhat = hyp_distances(sample_qhat,sample_qhat,1,params[u'c'])
    _, _, dist_pz_qhat = hyp_distances(sample_pz,sample_qhat,1,params[u'c'])

    # Median heuristic for the sigma^2 of Gaussian kernel
    l = 1500 # roughly half of full batch
    sigma2_k = tf.nn.top_k(
        tf.reshape(dist_pz_qhat, [-1]), l).values[l - 1]
    sigma2_k += tf.nn.top_k(
        tf.reshape(dist_qhat, [-1]), l).values[l - 1]

    #sigma2_k = 1.0
    #if params['verbose']:
    #    sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
    # Part (1)
    res1 = tf.exp( - dist_pz / 2. / sigma2_k)
    res1 = tf.multiply(res1, 1. - tf.eye(n)) # filter i=j pairs of samples
    res1 = tf.reduce_sum(res1) / (n * n - n)
    # Part (2)
    res2 = tf.exp( - dist_pz_qhat / 2. / sigma2_k)
    res2 = tf.reduce_sum(res2) / (n * n) / ns
    # Part (3)
    res3 = tf.exp( - dist_qhat / 2. / sigma2_k)
    res3 = tf.multiply(res3, 1. - tf.eye(n)) # filter i=j pairs of samples
    res3 = tf.reduce_sum(res3) / (n * n - n)
    stat = res1 - 2 * res2 + res3 

    #return stat, dist_qhat_prime ,res3, tf.boolean_mask(dist_qhat_prime,dist_qhat_prime<1.0)
    return stat, res1,res2,res3

def kld_loss(mu,sigma,d):
    Z = tf.reduce_sum(2*np.pi*tf.sqrt(np.pi/2)*sigma*tf.exp(sigma**2/2)*tf.erf(sigma/tf.sqrt(2.0)),axis=1)
    mu_norm = tf.norm(mu,2,axis=1)
    return tf.reduce_mean(mu_norm**2-mu_norm*(np.sqrt(2.0)*tf.lgamma((d+1)/2)/tf.lgamma(d/2))-tf.log(Z))
    
