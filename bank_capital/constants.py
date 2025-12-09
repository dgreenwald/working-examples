#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 13:53:02 2025

@author: dan
"""

# plot_vars = ['log_R_new', 'log_P_security', 
#         # 'log_required_capital_no_aoci_C', 
#         # 'log_required_capital_bank_C',
#         'dV_afs_bank_C',
#         'spread_at_issuance_C',
#         # 'log_y', 
#         # 'log_loans_new_C',
#         'spread_at_issuance_after_tax_C',
#         # 'log_debt_new', 
#         'log_inv', 'log_cash_new', 'D',
#         'log_debt_new_U', 'inv_U', 'log_cash_new_U', 'D_U',
#         'log_debt_new_C', 'inv_C', 'log_cash_new_C', 'D_C',
#         'capital_buffer_scaled_bank_C', 'log_deposits_new_bank_C', 'V_afs_bank_C', 'aoci_bank_C', 'loss_exposure_C',
#         'D_bank_C', 'log_deposits_new_C', 'loans_to_assets_bank_C', 'profit_bank_C', 'profit_to_assets_bank_C', 
#         'log_assets_bank_C', 'log_capital_bank_C',
#         'net_interest_income_bank_C', 'r_deposit_new',
#         ]

plot_vars = ['R_new', 'P_security', 
        # 'required_capital_no_aoci_C', 
        # 'required_capital_bank_C',
        'dV_afs_bank_C',
        'spread_at_issuance_C',
        # 'y', 
        # 'loans_new_C',
        'spread_at_issuance_after_tax_C',
        # 'debt_new', 
        'inv', 'cash_new', 'D',
        'debt_new_U', 'inv_U', 'cash_new_U', 'D_U',
        'debt_new_C', 'inv_C', 'cash_new_C', 'D_C',
        'capital_buffer_scaled_bank_C', 'deposits_new_bank_C', 'V_afs_bank_C', 'aoci_bank_C', 'loss_exposure_C',
        'D_bank_C', 'deposits_new_C', 'loans_to_assets_bank_C', 'profit_bank_C', 'profit_to_assets_bank_C', 
        'assets_bank_C', 'capital_bank_C',
        'net_interest_income_bank_C', 'r_deposit_new',
        #
        'Q_U', 'Q_C', 'net_outflow_financial_U', 'net_outflow_financial_C',
        'D_U', 'D_C', 'deposits_new_bank_C',
        'Om_principal_deposits_bank_C', 'Om_spread_deposits_bank_C', 'Lam_1_bank_C', 
        'spread_L_C',
        'spread_L_implied_C',
        ]

mod_labels_colors_styles_markers = {
    'baseline' : ('Baseline', 'firebrick', '-', '^'),
    'baseline_no_mtm' : ('Book Value', 'cornflowerblue', '-', 'o'),
    }

labels = {}
colors = {}
styles = {}
markers = {}

for key, this_data in mod_labels_colors_styles_markers.items():
    
    labels[key], colors[key], styles[key], markers[key] = this_data