import sys
import json
import argparse
import numpy as np

from psrqpy import QueryATNF
from .approximation import PulsarApproximation
from .pinn import PulsarPINN
from .utils import configure_logging

import warnings
warnings.filterwarnings('ignore')

def query_atnf(params, condition=None, max_pulsars=None):
    try:
        query = QueryATNF(params=params, condition=condition)
        table = query.table
        
        result = {
            'success': True,
            'num_pulsars': len(table),
            'params': params,
            'data': {}
        }
        
        for param in params:
            if param in table.colnames:
                data = table[param].data
                if isinstance(data, np.ndarray):
                    data = data.tolist()
                result['data'][param] = data
        
        if max_pulsars and len(table) > max_pulsars:
            for key in result['data']:
                result['data'][key] = result['data'][key][:max_pulsars]
            result['num_pulsars'] = max_pulsars
        
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def fit_polynomial(x_param, y_param, test_degree=5, log_x=True, log_y=True,
                   atnf_condition=None, verbose=False):
    try:
        configure_logging('ERROR')
        
        query = QueryATNF(params=[x_param, y_param], condition=atnf_condition)
        
        approx = PulsarApproximation(
            query=query,
            x_param=x_param,
            y_param=y_param,
            test_degree=test_degree,
            log_x=log_x,
            log_y=log_y
        )
        
        approx.fit_polynomial(verbose=verbose)
        metrics = approx.compute_metrics(verbose=False)
        
        result = {
            'success': True,
            'best_degree': int(approx.best_degree),
            'r2_scores': {int(k): float(v) for k, v in approx.r2_scores.items()},
            'coefficients': [float(c) for c in approx.coefficients],
            'intercept': float(approx.intercept),
            'metrics': {
                'r2': float(metrics['r2']),
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'chi2_reduced': float(metrics['chi2_reduced']),
                'n_samples': int(metrics['n_samples']),
                'n_params': int(metrics['n_params'])
            },
            'polynomial_expression': approx.get_polynomial_expression()
        }
        
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def analyze_data(x_param, y_param, atnf_condition=None):
    try:
        query = QueryATNF(params=[x_param, y_param], condition=atnf_condition)
        table = query.table
        
        x_vals = np.array(table[x_param], dtype=float)
        y_vals = np.array(table[y_param], dtype=float)
        
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        
        result = {
            'success': True,
            'total_pulsars': len(table),
            'valid_pulsars': len(x_vals),
            'x_stats': {
                'min': float(np.min(x_vals)),
                'max': float(np.max(x_vals)),
                'mean': float(np.mean(x_vals)),
                'median': float(np.median(x_vals)),
                'std': float(np.std(x_vals))
            },
            'y_stats': {
                'min': float(np.min(y_vals)),
                'max': float(np.max(y_vals)),
                'mean': float(np.mean(y_vals)),
                'median': float(np.median(y_vals)),
                'std': float(np.std(y_vals))
            }
        }
        
        return result
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='pulsarfitpy headless interface')
    parser.add_argument('command', choices=['query', 'fit', 'analyze'],
                       help='Command to execute')
    parser.add_argument('--json', type=str, required=True,
                       help='JSON string with parameters')
    
    args = parser.parse_args()
    
    try:
        params = json.loads(args.json)
    except json.JSONDecodeError as e:
        result = {'success': False, 'error': f'Invalid JSON: {str(e)}'}
        print(json.dumps(result))
        sys.exit(1)
    
    if args.command == 'query':
        result = query_atnf(
            params=params.get('params', ['P0', 'P1']),
            condition=params.get('condition'),
            max_pulsars=params.get('max_pulsars')
        )
    elif args.command == 'fit':
        result = fit_polynomial(
            x_param=params.get('x_param', 'P0'),
            y_param=params.get('y_param', 'P1'),
            test_degree=params.get('test_degree', 5),
            log_x=params.get('log_x', True),
            log_y=params.get('log_y', True),
            atnf_condition=params.get('condition'),
            verbose=params.get('verbose', False)
        )
    elif args.command == 'analyze':
        result = analyze_data(
            x_param=params.get('x_param', 'P0'),
            y_param=params.get('y_param', 'P1'),
            atnf_condition=params.get('condition')
        )
    else:
        result = {'success': False, 'error': f'Unknown command: {args.command}'}
    
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get('success', False) else 1)

if __name__ == '__main__':
    main()