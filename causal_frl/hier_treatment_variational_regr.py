import numpy as np
import itertools, functools
from ml_stuff.ml_stuff.hier_variational_regr import *
import python_utils.python_utils.caching as caching
import string
import fim

def simulate_data(L, N, d, x_var, mu, prec_val, lambda_val, delta, treat_prob):

#    z_ns_num = np.random.randint(0, L, N)
    z_ns_num = np.tile(np.arange(0, L, 1), 1 + ((N-1)/L))[0:N]
    z_ns = np.zeros((N,L))
    z_ns[np.arange(0,N,1), z_ns_num] = 1
    x_ns = np.random.multivariate_normal(np.zeros(d), np.eye(d) * x_var, N)
    #x_ns = np.ones((N,1))
    B_ls = np.random.multivariate_normal(mu, np.eye(d) / prec_val, L)
    T_ns = np.random.randint(0, 2, N)
    #T_ns = np.zeros(N)
    treatment_effect_ls = k_ls_f(L).dot(delta)
    treatment_boost_ns = T_ns * treatment_effect_ls[z_ns_num]
    lambda_ls = np.arange(lambda_val, lambda_val + L, 1)
#    lambda_ls = np.tile(lambda_val, L)
    B_ns = B_ls[z_ns_num,:]
    lambda_ns = lambda_ls[z_ns_num]
    y_ns = treatment_boost_ns + np.sum(x_ns * B_ns, axis=1) + np.random.normal(0, (1.0 / lambda_ns)**0.5)

    return x_ns, y_ns, z_ns, T_ns, B_ls


def init_variational_params(x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0):
    """
    
    """
    lambda_B = 1.0
    T_sigma = 1.0
    v = 10.0
    c = 10.0
    init_mu_delta = 1.0
    init_prec_delta = 1.0
    
    L = z_ns.shape[1]
    d = x_ns.shape[1]
    T = np.eye(d) * T_sigma
    m = np.ones(d)
    mu_B_ls = np.zeros((L,d))
    prec_B_ls = [np.eye(d) * lambda_B for l in xrange(L)]
    alpha_lambda_ls = np.ones(L)
    beta_lambda_ls = np.ones(L)*100.
    mu_delta = np.ones(L) * init_mu_delta
    prec_delta = np.diag(np.ones(L)) * init_prec_delta
    return v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta


def infer_variational_params(num_iter, x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta):
    """
    accepts hyperparameters, observed data, initial value of parameters
    """
    L = z_ns.shape[1]
    d = x_ns.shape[1]

    N = z_ns.shape[0]
    
#    print 'L: %d d: %d' % (L, d)
    
    k_ls = k_ls_f(L)
    k_ls_k_ls_T = k_ls_k_ls_T_f(L)
    eps = -0.00001

    old_evidence_val = None
    for i in xrange(num_iter):

        #print i, np.sum(mu_B_ls)
        v = v_0 + L
        c = c_0 + L
        if True:
            # update (mu,prec) params
            v = v_0 + L
            c = c_0 + L
            m = (L / (c_0 + L)) * (np.sum(mu_B_ls, axis=0) / L) + (c_0 / (c_0 + L)) * m_0
            T = np.linalg.inv(np.linalg.inv(T_0) + c_0*np.outer(m_0,m_0) + np.sum([E_B_l_B_l_T_f(mu_B_l, prec_B_l) for (mu_B_l, prec_B_l) in itertools.izip(mu_B_ls, prec_B_ls)], axis=0) - c*np.outer(m,m))
                
        if False:
            new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta)
            print '1 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), string.join(['%.3f' % d for d in mu_delta], sep=', '), 'L: %.6f' % new_evidence_val
            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val


        if True:
            # update delta params
            #pdb.set_trace()
#            print mu_delta
            prec_delta = np.sum([b * k_l_k_l_T for (b, k_l_k_l_T) in itertools.izip(E_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls) * np.sum(T_ns[:,np.newaxis] * z_ns, axis=0), k_ls_k_ls_T)], axis=0)
            try:
                mu_delta = np.linalg.inv(prec_delta).dot((np.sum((y_ns[:,np.newaxis] - x_ns.dot(E_B_ls_f(mu_B_ls, prec_B_ls).T)) * z_ns * T_ns[:,np.newaxis], axis=0).T * E_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls)).T.dot(k_ls).T)
                mu_delta = np.maximum(mu_delta, np.zeros(L))
            except:
                print prec_delta
                pdb.set_trace()
            #pdb.set_trace()


        if False:
            new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta)
            print '2 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), string.join(['%.3f' % d for d in mu_delta], sep=', '), 'L: %.6f' % new_evidence_val


            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val
            
            
        if True:
            # update B_ls params
            b_ns_ls = z_ns * (y_ns[:,np.newaxis] - T_ns[:,np.newaxis] * k_ls.dot(E_delta_f(mu_delta, prec_delta))[np.newaxis,:])
            for l in xrange(L):
                relevant_x_ns = x_ns[z_ns[:,l].astype(bool),:]
                prec_B_ls[l] = E_prec_f(v, T, c, m) + E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * relevant_x_ns.T.dot(relevant_x_ns)
#                prec_B_ls[l] = E_prec_f(v, T, c, m) + E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * np.diag(z_ns[:,l]).dot(x_ns).T.dot(np.diag(z_ns[:,l]).dot(x_ns))
#                print 'TESTING update mu_B_l before', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])
                #mu_B_ls[l] = np.linalg.inv(prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m) + E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) * x_ns.T.dot(z_ns[:,l] * y_ns))
#                import copy
                #np.linalg.inv(prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m) + b_ns_ls[:,l].T.dot(x_ns).T * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]))
#                print 'TESTING update mu_B_l middle', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])
                #mu_B_ls[l] = np.linalg.inv(prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m) + b_ns_ls[:,l].T.dot(relevant_x_ns).T * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]))
                #print prec_B_ls[l]
                #print 'DIFF', l, b_ns_ls[z_ns[:,l].astype(bool),l].T.dot(relevant_x_ns)
                #print l, 'lambda_l',  E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
                mu_B_ls[l] = np.linalg.inv(prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m) + b_ns_ls[z_ns[:,l].astype(bool),l].T.dot(relevant_x_ns).T * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]))
                #print 'mu_B_l', l, mu_B_ls[l]
                #first = E_prec_mu_f(v, T, c, m)
                #second = b_ns_ls[z_ns[:,l].astype(bool),l].T.dot(relevant_x_ns).T * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
                #print 'norms', np.linalg.norm(first), np.linalg.norm(second)
                #print 'multednorms', np.linalg.norm(np.linalg.inv(prec_B_ls[l]).dot(first)), np.linalg.norm(np.linalg.inv(prec_B_ls[l]).dot(second))
                #print 'rank', np.linalg.matrix_rank(relevant_x_ns)
                #sqB, sqRes, sqRank, sqS = np.linalg.lstsq(relevant_x_ns, b_ns_ls[z_ns[:,l].astype(bool),l])
                #pdb.set_trace()
                #print 'lstsq B', l, sqB
                #print 'lstsq res', l, sqRes
                #print 'x_ns sum', np.sum(relevant_x_ns, axis=0)
                #print 'ys', b_ns_ls[z_ns[:,l].astype(bool),l]
                #print 'ignore prior projection coeff:', np.linalg.inv(relevant_x_ns.T.dot(relevant_x_ns))#.dot(relevant_x_ns.T.dot(b_ns_ls[z_ns[:,l].astype(bool),l]))
#                print 'TESTING update mu_B_l after', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])

        if False:
            new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta)
            print '3 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), string.join(['%.3f' % d for d in mu_delta], sep=', '), 'L: %.6f' % new_evidence_val


            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val

        if True:
            # update lambda_ls params
            treatment_effect_ls = k_ls.dot(E_delta_f(mu_delta, prec_delta))
            cov_prediction_ns_ls = x_ns.dot(E_B_ls_f(mu_B_ls, prec_B_ls).T)
            E_delta_delta_T = E_delta_delta_T_f(mu_delta, prec_delta)
            for l in xrange(L):
                alpha_lambda_ls[l] = alpha_0 + 0.5 * np.sum(z_ns[:,l])
                E_B_l_B_l_T = E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])
                r_x_ns = x_ns[z_ns[:,l].astype(bool),:]
                r_y_ns = y_ns[z_ns[:,l].astype(bool)]
                r_T_ns = T_ns[z_ns[:,l].astype(bool)]
                beta_lambda_ls[l] = beta_0 + 0.5 * (np.sum(r_y_ns**2 - 2 * r_y_ns * r_x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l]))) + np.sum(r_x_ns.T.dot(r_x_ns) * E_B_l_B_l_T))
                #beta_lambda_ls[l] = beta_0 + z_ns[:,l].dot(0.5 * (y_ns**2 - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T) for x_n in x_ns])))
                #assert np.abs(beta_lambda_ls[l] - temp) < .0001
#                print 'TESTING lambda', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])
                temp = -2. * y_ns * T_ns * treatment_effect_ls[l] + 2. * T_ns * treatment_effect_ls[l] * cov_prediction_ns_ls[:,l] + T_ns * np.sum(np.outer(k_ls[l], k_ls[l]) * E_delta_delta_T)
                #cov_prediction_ns_l = r_x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l]).T)
                #temp2 = np.sum(0.5 * (-2. * r_y_ns * r_T_ns * treatment_effect_ls[l] + 2. * r_T_ns * treatment_effect_ls[l] * cov_prediction_ns_l + r_T_ns * np.sum(np.outer(k_ls[l], k_ls[l]) * E_delta_delta_T)))
                #assert np.abs(temp2 - z_ns[:,l].dot(0.5*temp)) < .001
                beta_lambda_ls[l] += z_ns[:,l].dot(0.5*temp)
                #beta_lambda_ls[l] += temp2
                
        if False:
            new_evidence_val = log_evidence_lower_bound(x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta)
            print '4 step %d' % i, 'mu_B_ls sum: %.3f' % np.sum(mu_B_ls), string.join(['%.3f' % d for d in mu_delta], sep=', '), 'L: %.6f' % new_evidence_val


            if not old_evidence_val is None:
                assert new_evidence_val - old_evidence_val > eps
            old_evidence_val = new_evidence_val
            
        if False:
            print 'v',v
            print 'T', T
            print 'c', c
            print 'm', m
            print 'mu_B_ls', mu_B_ls
            print 'prec_B_ls', prec_B_ls
            print 'alpha_lambda_ls', alpha_lambda_ls
            print 'beta_lambda_ls', beta_lambda_ls
            
    return v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta


def log_evidence_lower_bound(x_ns, y_ns, z_ns, T_ns, v_0, T_0, c_0, m_0, alpha_0, beta_0, r_0, s_0, v, T, c, m, mu_B_ls, prec_B_ls, alpha_lambda_ls, beta_lambda_ls, mu_delta, prec_delta):

#    print 'GOOD MU_B_ls', mu_B_ls
    
    L = z_ns.shape[1]
    d = x_ns.shape[1]
    
    val = 0
    debug_print(1, 'CALCULATING EVIDENCE')

    k_ls = k_ls_f(L)
    k_ls_k_ls_T = k_ls_k_ls_T_f(L)

    
    # p_data
    treatment_effect_ls = k_ls.dot(E_delta_f(mu_delta, prec_delta))
    cov_prediction_ns_ls = x_ns.dot(E_B_ls_f(mu_B_ls, prec_B_ls).T)
    E_delta_delta_T = E_delta_delta_T_f(mu_delta, prec_delta)
    for l in xrange(L):

        E_B_l_B_l_T = E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])
        r_x_ns = x_ns[z_ns[:,l].astype(bool),:]
        r_y_ns = y_ns[z_ns[:,l].astype(bool)]
        r_T_ns = T_ns[z_ns[:,l].astype(bool)]
        num = sum(z_ns[:,l])
        temp_new = num * (- d * np.log(2*np.pi) / 2 \
                             + 0.5 * E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]))
        temp_new_temp = np.sum(r_y_ns**2 \
                    - 2 * r_y_ns * r_x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])))
        temp_new_temp +=  np.sum(r_x_ns.T.dot(r_x_ns) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l]))
        temp_new_temp *= - 0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
        temp_new += temp_new_temp
                             
        #new = z_ns[:,l].dot(\
        #                     - d * np.log(2*np.pi) / 2 \
        #                     + 0.5 * E_log_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) \
        #                     - 0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) \
        #                     * (y_ns**2 \
        #                        - 2 * y_ns * x_ns.dot(E_B_l_f(mu_B_ls[l], prec_B_ls[l])) \
        #                        + np.array([np.sum(np.outer(x_n, x_n) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l])) for x_n in x_ns])))
        #print 'TESTING data', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])
        #print temp_new, new, 'compare'
        #assert np.abs(temp_new - new) < .001
        temp = -2. * y_ns * T_ns * treatment_effect_ls[l] + 2. * T_ns * treatment_effect_ls[l] * cov_prediction_ns_ls[:,l] + T_ns * np.sum(np.outer(k_ls[l], k_ls[l]) * E_delta_delta_T)
        temp_new += z_ns[:,l].dot(-0.5 * E_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l]) *temp) 
        val += temp_new
        
    # p_mu_prec
    new = -1. * wishart_log_Z_f(v_0, T_0) \
      + ((v_0-d-1)/2.) * E_log_det_prec_f(v, T, c, m) \
      - 0.5 * np.sum(np.linalg.inv(T_0) * E_prec_f(v, T, c, m)) \
      - (d/2.)*np.log(2*np.pi) \
      + 0.5 * (E_log_det_prec_f(v, T, c, m) + np.log(c_0))\
      - (c_0/2.) * np.sum(np.outer(m_0, m_0) * E_prec_f(v, T, c, m)) \
      - (c_0/2.) * E_trace_prec_mu_mu_T_f(v, T, c, m) \
      + c_0 * E_prec_mu_f(v, T, c, m).T.dot(m_0)
    #print new
    val += new

    # p_B_ls
    new = (-L*d/2.) * np.log(2*np.pi) \
      + (L/2.) * E_log_det_prec_f(v, T, c, m) \
      - (L/2.) * E_trace_prec_mu_mu_T_f(v, T, c, m)
    for l in xrange(L):
        new += -0.5 * np.sum(E_prec_f(v, T, c, m) * E_B_l_B_l_T_f(mu_B_ls[l], prec_B_ls[l]))
        new += E_B_l_f(mu_B_ls[l], prec_B_ls[l]).dot(E_prec_mu_f(v, T, c, m))
        #print 'TESTING B_l', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])

    #print new
    val += new
        
    # p_lambda_ls
    new = L * (alpha_0*np.log(beta_0) - scipy.special.gamma(alpha_0)) \
      + (alpha_0-1) * np.sum(E_log_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls)) \
      - beta_0 * np.sum(E_lambda_ls_f(alpha_lambda_ls, beta_lambda_ls))
    #print new
    val += new
    
    # H_mu_prec
    new = H_mu_prec_f(v, T, c, m)
    #print new
    val += new

    # H_B_ls
    for l in xrange(L):
        new = H_B_l_f(mu_B_ls[l], prec_B_ls[l])
#        print new, val, 'ggggg'
        val += new
        debug_print(1, 'update: %.2f H_B_ls' % new)
        debug_print(1, 'evidence: %.2f H_B_ls' % val)
        
    # H_lambda_ls
    for l in xrange(L):
        new = H_lambda_l_f(alpha_lambda_ls[l], beta_lambda_ls[l])
        #print 'TESTING H_lambda_l', l, E_B_l_f(mu_B_ls[l], prec_B_ls[l])
        val += new

    # H_delta
    new = H_B_l_f(mu_delta, prec_delta)
    #print 'delta entropy', new
    val += new
        
    return val
    
    
class model_fitter(object):
    """
    *(x_ns,T_ns),y_ns* -> M
    """
    def __call__(self, (x_ns, T_ns), y_ns):
        pass


class enumerate_all_model_fitter(object):

    def __init__(self, model_iterator):
        self.model_iterator = model_iterator
    
    def __call__(self, (x_ns, T_ns), y_ns):
        pass

from collections import namedtuple

class p_theta(namedtuple('p_theta', ['v_0', 'T_0', 'c_0', 'm_0', 'alpha_0', 'beta_0', 'r_0', 's_0'])):
    pass


class theta(namedtuple('theta', ['mu', 'prec', 'B_ls', 'lambda_ls', 'delta'])):
    pass


class rule(object):

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        #print repr(self), repr(other), repr(self) == repr(other)
        return repr(self) == repr(other)
    
    def __repr__(self):
        if self.feat_name is None:
            return '%d_%.2f_%d' % (self.k, self.cutoff, self.sign)
        else:
            return '%s_%.2f_%d' % (self.feat_name, self.cutoff, self.sign)
    
    def __init__(self, k, cutoff, sign, feat_name=None):
        self.k, self.cutoff, self.sign = k, cutoff, sign
        self.feat_name = feat_name

    def __call__(self, x):
        if self.sign == 1:
            return x[self.k] > self.cutoff
        elif self.sign == -1:
            return not (x[self.k] > self.cutoff)
        assert False
        return bool(self.sign * int(x[self.k] > self.cutoff))
    

class binary_rule(rule):

    def __init__(self, idx, sign=1, feat_name=None):
        self.idx = idx
        self.sign = sign
        self.feat_name = feat_name
        
    def __repr__(self):
        if self.feat_name is None:
            return 'bin%s_%d' % (str(self.idx), self.sign)
        else:
            return 'bin%s_%d' % (self.feat_name, self.sign)

    def __call__(self, x):
        try:
            iter(self.idx)
        except TypeError:
            return x[self.idx] == self.sign
        else:
            return np.sum([x[i] == self.sign for i in self.idx]) == len(self.idx)


class fpgrowth_miner_f(object):

    def __init__(self, supp, zmax):
        self.supp, self.zmax = supp, zmax

    def __call__(self, x_ns, feat_names = None):
        import pandas as pd
        def which_are_1(v):
            return list(pd.Series(range(len(v)))[map(bool,v)])
        feat_names = np.array(feat_names)
        length = float(x_ns.shape[0])
        raw = fim.fpgrowth([which_are_1(x_n) for x_n in x_ns], supp = self.supp, zmax = self.zmax)
        return [binary_rule(list(r),1,feat_names[list(r)]) for (r,s) in raw]
        
class mem_address_rule(object):

    def __init__(self, idx, total):
        self.idx, self.total = idx, total

    def __call__(self, x):
        return hash(np.sum(x)) % self.total == self.idx
    

class default_rule(object):

    def __repr__(self):
        return 'default'
    
    def __call__(self, x):
        return True

    
def subsequence_iterator(s, L=None):
    if L is None:
        L = len(s)
    return list(itertools.chain(*itertools.imap(functools.partial(itertools.permutations,s),xrange(1,L+1))))

    
class p_y_ns_given_theta(object):

    def __init__(self, rule_f_ls):
        self.rule_f_ls = rule_f_ls

    def sample(self, theta, (x_ns, T_ns)):
        N = len(x_ns)
        L = len(self.rule_f_ls)
        z_ns_num = self.get_z_ns_num(x_ns)
        treatment_effect_ls = k_ls_f(L).dot(theta.delta)
        treatment_boost_ns = T_ns * treatment_effect_ls[z_ns_num]
        B_ns = theta.B_ls[z_ns_num,:]
        lambda_ns = theta.lambda_ls[z_ns_num]
        y_ns = treatment_boost_ns + np.sum(x_ns * B_ns, axis=1) + np.random.normal(0, (1.0 / lambda_ns)**0.5)
        return y_ns
        
    def get_z_ns_num(self, x_ns):
        z_ns_num = np.zeros(len(x_ns),dtype=int) * -1
        for (n,x_n) in enumerate(x_ns):
            for (l,rule_f_l) in enumerate(self.rule_f_ls):
                if rule_f_l(x_n):
                    z_ns_num[n] = int(l)
                    break
        assert np.sum(z_ns_num==-1) == 0
        return z_ns_num
        
    def get_z_ns(self, x_ns):
        # fix
        z_ns_num = self.get_z_ns_num(x_ns)
        N = len(x_ns)
        L = len(self.rule_f_ls)
        z_ns = np.zeros((N,L))
        z_ns[np.arange(0,N,1), z_ns_num] = 1
        return z_ns
    

class model(object):

    def __repr__(self):
        return repr(self.p_y_ns_given_theta.rule_f_ls)

    @property
    def rule_fs(self):
        return self.p_y_ns_given_theta.rule_f_ls[0:-1]
    
    def __init__(self, p_theta, p_y_ns_given_theta):
        self.p_theta, self.p_y_ns_given_theta = p_theta, p_y_ns_given_theta

    @classmethod
    def model_from_p_theta_and_rule_fs(cls, p_theta, rule_fs):
        return cls(p_theta, p_y_ns_given_theta(rule_fs + [default_rule()]))

    def check_treatment_effects(self, mu_B_ls, (x_ns,T_ns), y_ns):
        z_ns_num = self.p_y_ns_given_theta.get_z_ns_num(x_ns)
        N = y_ns.shape[0]
        L = mu_B_ls.shape[0]
        effect_ns = (y_ns[:,np.newaxis] - x_ns.dot(mu_B_ls.T))[np.arange(0,N,1),z_ns_num]
        for l in xrange(L):
            print l, np.mean(effect_ns[z_ns_num==l])
            
class get_posterior_f(object):

    def __init__(self, num_iter):
        self.num_iter = num_iter
    
    def __call__(self, model, (x_ns, T_ns), y_ns):
        z_ns = model.p_y_ns_given_theta.get_z_ns(x_ns)
        variational_params = init_variational_params(x_ns, y_ns, z_ns, T_ns, *model.p_theta)
        return infer_variational_params(self.num_iter, x_ns, y_ns, z_ns, T_ns, *(list(model.p_theta)+list(variational_params)))


class get_evidence_f(object):

    def __init__(self, get_posterior_f, get_evidence_lower_bound_f):
        self.get_posterior_f, self.get_evidence_lower_bound_f = get_posterior_f, get_evidence_lower_bound_f

    #@caching.default_cache_method_decorator()
    def __call__(self, model, (x_ns, T_ns), y_ns):
        z_ns = model.p_y_ns_given_theta.get_z_ns(x_ns)
        print np.sum(z_ns,axis=0)
        variational_params = self.get_posterior_f(model, (x_ns, T_ns), y_ns)
        return self.get_evidence_lower_bound_f(x_ns, y_ns, z_ns, T_ns, *(list(model.p_theta)+list(variational_params)))


class take_best_model_fitter(object):

    def __init__(self, fitters):
        self.fitters = fitters

    def __call__(self, (x_ns, T_ns), y_ns):
        results = []
        for fitter in self.fitters:
            model, evidence = fitter((x_ns, T_ns), y_ns)
            results.append([model, evidence])
        best_model, best_evidence = max(results, key = lambda (model,evidence):evidence)
        return (best_model, best_evidence), results
            
class simulated_annealing_model_fitter(object):

    def __init__(self, rule_fs, p_theta, num_iters, temperature, get_evidence_f, reject_f):
        """
        rule_fs should not contain default rule
        """
        self.rule_fs, self.p_theta, self.num_iters, self.temperature, self.get_evidence_f, self.reject_f = rule_fs, p_theta, num_iters, temperature, get_evidence_f, reject_f

    def model_from_rule_fs(self, rule_fs):
        """
        rule_fs does not contain default rule
        """
        return model.model_from_p_theta_and_rule_fs(self.p_theta, rule_fs)

    def unused_rules(self, candidate_model):
        ans = list(set(self.rule_fs) - set(candidate_model.rule_fs))
        #print self.rule_fs, candidate_model.rule_fs
        return ans
    
    def successor(self, old_model):
        # possible moves are add(at least 1 in unused), remove(at least 1 in model), swap(at least 2 in model), replace(at least 1 in model)
        import copy
        while 1:
            which = np.random.randint(0,4)
            if which == 0:
                # add
                #print 'add'
                unused_rules = self.unused_rules(old_model)
                #print 'unused_rules', unused_rules
                if len(unused_rules) > 0:
                    new_rule = np.random.choice(unused_rules)
                    new_position = np.random.randint(0, len(old_model.rule_fs)+1)
                    new_rule_fs = copy.deepcopy(old_model.rule_fs)
                    new_rule_fs.insert(new_position, new_rule)
                    break
            elif which == 1:
                # remove
                #print 'remove'
                old_rule_fs = old_model.rule_fs
                if len(old_rule_fs) > 0:
                    remove_position = np.random.randint(0, len(old_rule_fs))
                    new_rule_fs = copy.deepcopy(old_rule_fs)
                    new_rule_fs.pop(remove_position)
                    break
            elif which == 2:
                # swap
                #print 'swap'
                old_rule_fs = old_model.rule_fs
                if len(old_rule_fs) >= 2:
                    position_1, position_2 = np.random.choice(range(len(old_rule_fs)), 2, replace=False)
                    new_rule_fs = copy.deepcopy(old_rule_fs)
                    temp = new_rule_fs[position_1]
                    new_rule_fs[position_1] = new_rule_fs[position_2]
                    new_rule_fs[position_2] = temp
                    break
            elif which == 3:
                # replace
                #print 'replace'
                old_rule_fs = old_model.rule_fs
                unused_rules = self.unused_rules(old_model)
                if len(old_rule_fs) > 0 and len(unused_rules) > 0:
                    new_rule = np.random.choice(unused_rules)
                    new_position = np.random.randint(0, len(old_rule_fs))
                    new_rule_fs = copy.deepcopy(old_rule_fs)
                    new_rule_fs[new_position] = new_rule
                    break
        #print new_rule_fs
        return self.model_from_rule_fs(new_rule_fs)
    
    def __call__(self, (x_ns, T_ns), y_ns):
        best_model, best_evidence = None, None
        while 1:
            old_model = self.model_from_rule_fs(list(np.random.choice(self.rule_fs,min(max(len(self.rule_fs)/2,1),3),replace=False)))
            if not self.reject_f(old_model, (x_ns,T_ns), y_ns):
                break
        print 'check initial', self.reject_f(old_model, (x_ns,T_ns), y_ns)
        i = 0
        #for i in xrange(self.num_iters):
        while i < self.num_iters:
            new_model = self.successor(old_model)
            if i % 10 == 0:
                print '\t\t\t\t\t\tstep', i
            if not self.reject_f(new_model,(x_ns,T_ns), y_ns):
                #print 'old', old_model
                old_evidence = self.get_evidence_f(old_model,(x_ns, T_ns), y_ns)
                #print 'new', new_model
                new_evidence = self.get_evidence_f(new_model,(x_ns, T_ns), y_ns)
                print old_evidence, old_model, new_evidence, new_model
                if best_evidence is None or new_evidence > old_evidence:
                    best_model = new_model
                    best_evidence = new_evidence
                if np.random.random() < np.exp(new_evidence - old_evidence) / self.temperature:
                    old_model = new_model
                i += 1
            else:
                print 'ff', self.reject_f(new_model,(x_ns,T_ns), y_ns), new_model
                #pdb.set_trace()
        return best_model, best_evidence
#        return old_model, old_evidence
            
