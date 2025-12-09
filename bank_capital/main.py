#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Nov 25 15:52:50 2025

@author: dan
'''

from danare.solvers import deterministic, linear
# from danare import Model
from danare.solvers.det_spec import DetSpec
from danare.plot import plot_deterministic_results, plot_model_irfs
# from danare.settings import get_settings

from danare.modspec import ModSpec

import constants as K

# import jax
# import numpy as np

# from danare.model. model import ModelBlock

# def update_model(mod_spec):

from spec import create_model, update_model

# if __name__ == '__main__':   
    
base_mod_specs = [
    ['baseline'],
    # ['baseline', 'no_mtm'],
    ]

run_list = [
    'steady',
    # 'irfs',
    # 'deterministic',
    # 'compare_irfs',
    ]

mods = []

for base_mod_spec in base_mod_specs:
    
    mod_spec = ModSpec(base_mod_spec)
    
    flags = update_model(mod_spec)
    
    mod = create_model(flags, label=mod_spec.label)
    
    if 'steady' in run_list:
        
        mod.solve_steady(calibrate=True, save=True, load_initial_guess=True)
    
    if 'irfs' in run_list:
        
        mod.linearize()
        mod.plot_linear_irfs(include_list=K.plot_vars, plot_type='pdf',
                             max_per_page=16, n_per_row=4)
        
    if 'deterministic' in run_list:
        
        det_spec = DetSpec()
        # det_spec.add_regime(0, preset_par_regime={'bet_S' : 0.99})
        det_spec.add_shock(0, 'Z_til', 0, 0.01)
        
        Nt = 100
        res = linear.solve_sequence_linear(det_spec, mod, Nt)
        # res = deterministic.solve_sequence(det_spec, mod, Nt)
        
        # plot_dir = '/home/dan/Dropbox/output/y14/danare/bank_capital/plots/deterministic'
        plot_deterministic_results(
            [res], include_list=K.plot_vars, max_per_page=16, 
            n_per_row=4, plot_type='pdf'
            )
        
    mods.append(mod)
    
if 'compare_irfs' in run_list:
    
    plot_model_irfs(mods, 'tv_security_shock', include_list=K.plot_vars,
                    max_per_page=16, n_per_row=4, 
                    var_titles=K.labels, group_colors=K.colors, 
                    marker_styles=K.markers, group_styles=K.styles,
                    )
    
        
        
        
        
        
        
        