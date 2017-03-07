import numpy as np
from numpy.random import multivariate_normal as mvn
from itertools import izip
import math

class GeneticOptimizer(object):

    def __init__(self, crossover_func, mutate_func, selection_func,
                 checker_func=None, prob_cx=0.6, popsize=None, 
                 elitist=False, verbose=False):
        self.crossover = crossover_func
        self.mutate = mutate_func
        self.selection = selection_func
        self.prob_cx = prob_cx
        self.elitist = elitist
        self.population = []
        self.popsize = popsize
        self.checker = checker_func
        if popsize is not None and popsize % 2 != 0:
            raise ValueError('popsize must be even')
        self.verbose = verbose

    def initialize(self, initial_pop):
        if self.popsize is None and len(initial_pop) % 2 != 0:
            raise ValueError('Requires an even-numbered population')
        self.population = initial_pop

    def ask(self):
        if self.verbose:
            msg = 'Referring population:\n'
            for i, pop in enumerate(self.population):
                msg += '\tMember %d: %s\n' % (i, str(pop))
            print msg[:-2]
        return self.children

    def tell(self, fitness):
        if len(fitness) != len(self.population):
            raise ValueError('Expected %d fitness values, got %d',
                             (len(self.population), len(fitness)))

        if self.verbose:
            msg = 'Received:\n'
            for i, pop, fit in izip(range(len(self.population)), self.population, fitness):
                msg += '\tChild %d: %s fit: %f\n' % (i, str(pop), fit)
            print msg[:-2]

        # Record population best
        if self.elitist:
            ranked = zip(fitness, self.population)
            ranked.sort(key=lambda x: x[0], reverse=True)
            elite = ranked[0][1]

        # Select popsize
        if self.popsize is None:
            N = len(self.population)
        else:
            N = self.popsize

        # Pick new population parents
        parents = self.selection(N, self.pop_fitness)
        crossover_probs = np.random.rand(N/2)
        children = []
        for i in range(N/2):
            a_ind = parents[2*i]
            b_ind = parents[2*i+1]
            par_a = self.population[a_ind]
            par_b = self.population[b_ind]
            if crossover_probs[i] < self.prob_cx:
                child_a, child_b = self.crossover(par_a, par_b)
                if self.verbose:
                    msg = 'Crossover:\n'
            else:
                child_a = par_a
                child_b = par_b
                if self.verbose:
                    msg = 'No crossover:\n'
            
            if self.verbose:
                msg += '\tParent %d: %s fit:%f\n' % (a_ind, str(par_a), fitness[a_ind])
                msg += '\tParent %d: %s fit:%f\n' % (b_ind, str(par_b), fitness[b_ind])
                print msg

            # Apply mutations
            child_a = self.mutate(child_a)
            child_b = self.mutate(child_b)
            if self.checker is not None:
                child_a = self.checker(child_a)
                child_b = self.checker(child_b)
            if self.verbose:
                msg = 'Mutation:\n'
                msg += '\tChild A: %s\n' % str(child_a)
                msg += '\tChild B: %s' % str(child_b)
                print msg
            children.append(child_a)
            children.append(child_b)

        if self.elitist:
            children[-1] = elite
        self.population = children

def uniform_crossover(a, b, cx_rate=0.5):
    '''Perform crossover (breeding) on two iterable objects.
    '''
    if len(a) != len(b):
        raise ValueError('Genes must be of same length')

    a_picks = np.random.rand(len(a)) < cx_rate
    out_a = [ai if p else bi for p, ai, bi in izip(a_picks, a, b)]
    out_b = [bi if p else ai for p, ai, bi in izip(a_picks, a, b)]
    return out_a, out_b

def gaussian_mutate(a, cov):
    '''Perform Gaussian mutation on a vector gene.
    '''
    N = len(a)

    if np.iterable(cov):
        cov = np.asarray(cov)
    elif type(cov) is float or type(cov) is int:
        cov = cov * np.identity(N)
    else:
        raise ValueError('cov must be numeric')

    if len(cov.shape) < 2:
        cov = np.diag(cov)

    return a + mvn(mean=np.zeros(N), cov=cov)

def categorical_mutate(a, categories, weights=None):
    '''Mutates a categorical gene.
    '''
    if not a in categories:
        raise ValueError('Gene is not in category list')

    if weights is None:
        weights = np.ones(len(categories))
    else:
        weights = np.asarray(weights)
    
    sample_weights = [w if c != a else 0 for w,c in izip(weights, categories)]
    sample_weights = np.asarray(sample_weights)
    sample_weights = sample_weights / np.sum(sample_weights)
    return np.random.choice(categories, p=sample_weights)

def low_variance_selection(N, weights, comb_width=0.9):
    if comb_width > 1.0 or comb_width < 0.0:
        raise ValueError('comb_width must be between 0 and 1.')
    
    inds = range(len(weights))
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    
    # Special case of N=1
    if N == 1:
        return np.random.choice(inds, p=weights)
        
    cum_weights = np.cumsum(weights)
    shift = np.random.uniform(0, 1.0-comb_width)
    comb = np.linspace(start=0, stop=comb_width, num=N) + shift
    print comb
    picks = [np.argmax(cum_weights > c) for c in comb]
    return picks
    
def tournament_selection(N, weights, k_size=None):
    num_competitors = len(weights)

    if k_size is None:
        k_size = max(math.ceil(0.5 * num_competitors), 1)

    ranks = list(enumerate(weights))
    def select():
        # TODO Can make more efficient with cumprod?
        best_ind = None
        best_fit = float('-inf')
        for i in range(k_size):
            ind = np.random.random_integers(low=0, high=num_competitors-1)
            if weights[ind] > best_fit:
                best_ind = ind
                best_fit = weights[ind]
        return best_ind
    return [select() for i in range(N)]