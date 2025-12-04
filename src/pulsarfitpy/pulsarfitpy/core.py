import numpy as np

from .approximation import PulsarApproximation
from .pinn import PulsarPINN
from .utils import configure_logging
from psrqpy import QueryATNF

def create_approximation(x_param, y_param, test_degree=5, log_x=True, log_y=True, 
                        atnf_params=None, atnf_condition=None):
    if atnf_params is None:
        atnf_params = [x_param, y_param]
    
    query = QueryATNF(params=atnf_params, condition=atnf_condition)
    
    return PulsarApproximation(
        query=query,
        x_param=x_param,
        y_param=y_param,
        test_degree=test_degree,
        log_x=log_x,
        log_y=log_y
    )

def create_pinn(equation, x_param, y_param, domain, backend='deepxde',
                nn_architecture=None, device='cpu', atnf_params=None, 
                atnf_condition=None):
    if atnf_params is None:
        atnf_params = [x_param, y_param]
    
    query = QueryATNF(params=atnf_params, condition=atnf_condition)
    
    return PulsarPINN(
        equation=equation,
        atnf_query=query,
        domain=domain,
        backend=backend,
        nn_architecture=nn_architecture,
        device=device
    )

def quick_fit(x_param, y_param, test_degree=5, log_x=True, log_y=True,
              atnf_condition=None, show_plots=True, verbose=True):
    approx = create_approximation(
        x_param=x_param,
        y_param=y_param,
        test_degree=test_degree,
        log_x=log_x,
        log_y=log_y,
        atnf_condition=atnf_condition
    )
    
    approx.fit_polynomial(verbose=verbose)
    metrics = approx.compute_metrics(verbose=verbose)
    
    if show_plots:
        approx.plot_combined_analysis()
    
    return approx, metrics

__all__ = [
    'create_approximation',
    'create_pinn', 
    'quick_fit',
    'PulsarApproximation',
    'PulsarPINN',
    'configure_logging',
    'QueryATNF'
]