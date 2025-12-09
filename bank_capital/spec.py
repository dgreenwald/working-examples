#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Nov 25 15:52:50 2025

@author: dan
'''

# from danare.solvers import deterministic
from danare import Model
# from danare.solvers.det_spec import DetSpec
# from danare.plot import plot_deterministic_results
# from danare.settings import get_settings

# import jax
import numpy as np

from danare.model.model import ModelBlock

def update_model(mod_spec):
    
    flags = {}
    
    if 'no_mtm' in mod_spec:
        flags['mark_to_market'] = False
        
    return flags

def get_agent_type(agent):
    
    if any([num in agent for num in ['0', '1']]):
        agent_type = agent.replace('0', '').replace('1', '')
        is_main_type = False
    else:
        agent_type = agent
        is_main_type = True
        
    return agent_type, is_main_type

def preference_block(
    *,
    consumption_var='c',
    nominal: bool = True,
    preferences: str = 'crra',
) -> ModelBlock:
    """Standard CRRA preference block with optional housing/labor."""
    
    block = ModelBlock()
    
    if preferences == 'crra':
        block.rules['intermediate'] += [
            ('uc_AGENT', f'{consumption_var}_AGENT ** (-psi_ATYPE)'),
            ]
    elif preferences == 'cara':
        block.rules['intermediate'] += [
            ('uc_AGENT', f'np.exp(-psi_ATYPE * {consumption_var}_AGENT)'),
            ]
    else:
        raise Exception
    
    block.rules['intermediate'] += [
        ('Lam_1_AGENT', 'uc_AGENT'),
        ('Lam_0_AGENT', 'uc_AGENT / bet_ATYPE'),
        ]
    
    if nominal:
        block.rules['intermediate'] += [
            ('Lam_1_nom_AGENT', 'Lam_1_AGENT / pi'),
        ]
    
    return block

def investment_block() -> ModelBlock:
    """Investment frictions"""
    
    block = ModelBlock()
    
    block.rules['intermediate'] += [
        ('tv_inv_efficiency', '1.0'),
        ('Phi_inv_AGENT', 'tv_inv_efficiency * (inv_cost_0_ATYPE'
         ' + inv_cost_1_ATYPE * pow(inv_rate_AGENT, 1.0 - inv_xi)'
         ' / (1.0 - inv_xi))'),
        ('d_Phi_inv_AGENT', 'inv_cost_1_ATYPE * tv_inv_efficiency * pow(inv_rate_AGENT, -inv_xi)'),
        ('Q_AGENT', '1.0 / d_Phi_inv_AGENT'),
        ('Q_bar_AGENT', 'Q_AGENT + (Q_AGENT * Phi_inv_AGENT - inv_rate_AGENT) / (1.0 - delta)'),
        ('inv_cost_1_AGENT', 'delta ** inv_xi'),
        ('inv_cost_0_AGENT', 'delta - inv_cost_1_AGENT * (delta ** (1.0 - inv_xi)) / (1.0 - inv_xi)'),
        ]

    # (no optimality eq here; inv_rate comes from firm FOC)
    
    return block

def cash_block() -> ModelBlock:
    """Cash in the utility and optimality conditions"""
    
    block = ModelBlock()
    
    block.rules['intermediate'] += [
        ('du_cash_AGENT', 'cash_mult_ATYPE * (cash_new_AGENT ** (-cash_exp))'),
        ('net_outflow_cash_AGENT', 'cash_new_AGENT - (1.0 + r_cash_lag) * cash_AGENT / pi'),
        ]
    
    block.rules['expectations'] += [
        ('E_Om_cash_AGENT', '(Lam_1_nom_AGENT_NEXT / Lam_0_AGENT) * (1.0 + r_cash_lag_NEXT)'),
        ]
    
    block.rules['optimality'] += [
        ('cash_new_AGENT', '(1.0 - du_cash_AGENT) - E_Om_cash_AGENT'),
        ]
    
    block.rules['transition'] += [
        ('cash_AGENT', 'cash_new_AGENT'),
        ]
    
    block.rules['analytical_steady'] += [
        ('cash_AGENT', 'cash_new_AGENT'),
        ]
    
    return block
    

def firm_block(
    *,
    nominal: bool = True,
    fixed_wage: bool = True,
    bonds_only: bool = True, # TODO
    is_main_type: bool = False,
    **kwargs
) -> ModelBlock:
    """Firm behavior"""
    
    block = ModelBlock()
    
    block.rules['intermediate'] += [
        ('lab_AGENT', 'lab_bar_AGENT'),
        ('K_new_AGENT', 'inv_AGENT + (1.0 - delta) * K_AGENT'),
        ('y_AGENT', 'Z * (K_AGENT ** alp) * (lab_AGENT ** (1.0 - alp))'),
        ('fk_AGENT', 'alp * y_AGENT / K_AGENT'),
        ('flab_AGENT', '(1.0 - alp) * y_AGENT / lab_AGENT'),
        ('K_AGENT', 'np.exp(log_K_AGENT)'),
        ('wage_bill_AGENT', 'flab_AGENT * lab_AGENT'),
        ('inv_AGENT', 'inv_rate_AGENT * K_AGENT'),
        ('Rk_Q_avg_AGENT', 'after_tax_corp * x_AGENT + (1.0 - after_tax_corp * delta) * Q_bar_AGENT'),
        ('Rk_Q_marg_AGENT', 'after_tax_corp * fk_AGENT + (1.0 - after_tax_corp * delta) * Q_bar_AGENT'),
        ('bet_pi_AGENT', 'bet_AGENT / pi_bar'),
        #
        # Profit measures
        ('ebitda_AGENT', 'y_AGENT - wage_bill_AGENT'),
        ('x_AGENT', 'ebitda_AGENT / K_AGENT'),
        #
        # TODO: to be updated
        # ('D_AGENT', 'ebitda_AGENT - inv_AGENT'),
        ('D_AGENT', 'Rk_Q_avg_AGENT * K_AGENT - net_outflow_financial_AGENT - K_new_AGENT * Q_AGENT'),
        #
        # TODO: temporary code
        # ('debt_new_AGENT', '0.3 * K_new_AGENT'),
        #
        ('avg_principal_flow_debt_AGENT', 'frac_debt_maturing + after_tax_corp * r_lag + xi_AGENT'),
        ('marg_principal_flow_debt_AGENT', 'avg_principal_flow_debt_AGENT + d_xi_d_om_AGENT * d_om_d_B_AGENT * debt_AGENT'),
        ('marg_spread_flow_debt_AGENT', 'after_tax_corp'),
        #
        # Quantities
        ('principal_payments_AGENT', 'avg_principal_flow_debt_AGENT * debt_AGENT'), # TODO: should we be deducting xi_AGENT?
        #
        # TODO: this is potentially redundant with loan_issuance_BANK
        ('debt_issuance_AGENT', 'debt_new_AGENT - frac_debt_remaining * debt_AGENT / pi'),
        ('spread_payments_new_AGENT', 'debt_issuance_AGENT * spread_at_issuance_AGENT + frac_debt_remaining * spread_payments_AGENT / pi'),
        ('liabilities_new_AGENT', 'debt_new_AGENT + other_liabilities_ATYPE'),
        ('loans_new_AGENT', '(1.0 - bond_weight_ATYPE) * debt_new_AGENT'),
        ('loans_AGENT', '(1.0 - bond_weight_ATYPE) * debt_AGENT'),
        #
        # Net flows
        ('net_outflow_debt_AGENT', '(principal_payments_AGENT + spread_payments_AGENT) / pi - debt_issuance_AGENT'),
        ('net_outflow_financial_AGENT', 'net_outflow_debt_AGENT + net_outflow_cash_AGENT'),
    ]

    block.rules['expectations'] += [
        ('E_Om_K_AGENT', '(Lam_1_AGENT_NEXT / Lam_0_AGENT) * (Rk_Q_marg_AGENT_NEXT + (1.0 - rho_X) * Psi_AGENT_NEXT * fk_AGENT_NEXT)'),
    ]

    block.rules['transition'] += [
        ('log_K_AGENT', 'np.log(K_new_AGENT)'),
        ('debt_AGENT', 'debt_new_AGENT'),
        ('spread_payments_AGENT', 'spread_payments_new_AGENT'),
    ]

    block.rules['optimality'] += [
        ('inv_rate_AGENT', 'E_Om_K_AGENT - Q_AGENT'),
        ('debt_new_AGENT', '1.0 - Om_principal_debt_AGENT - Om_spread_debt_AGENT * spread_at_issuance_AGENT')
    ]
    
    block.rules['analytical_steady'] += [
        ('debt_AGENT', 'debt_new_AGENT'),
        # ('spread_payments_AGENT', 'spread_payments_new_AGENT'),
        ]
    
    if bonds_only:
        
        block.rules['intermediate'] += [
            ('spread_at_issuance_AGENT', 'spread_B_bar'),
            ]
        
    else:
        
        block.rules['intermediate'] += [
            ('spread_at_issuance_AGENT', 'bond_weight_ATYPE * spread_B_bar + (1.0 - bond_weight_ATYPE) * spread_L_AGENT'),
            ]
        
    if is_main_type:
        
        block.rules['intermediate'] += [
            ('mu_om_AGENT', '-0.5 * (sig_om_AGENT ** 2)'),
            ('sig_om_AGENT', 'np.exp(log_sig_om_AGENT)'),
            ('kap_viol_AGENT', 'np.exp(log_kap_viol_AGENT)'),
            ]
        
        block.rules['calibration'] += [
            ('w_bar_AGENT', 'flab_AGENT - w_bar_AGENT'),
            ('cash_mult_AGENT', 'cash_new_AGENT / K_new_AGENT - cash_to_K_target_ATYPE'),
            ('other_liabilities_AGENT', 'debt_new_AGENT / liabilities_new_AGENT - debt_to_liabilities')
            ]
    
    return block

def covenant_block() -> ModelBlock:
    """Imposes debt covenants"""
    
    block = ModelBlock()
    
    block.rules['transition'] += [
        ('ebitda_smooth_AGENT', 'ebitda_smooth_new_AGENT'),
        ]
    
    block.rules['intermediate'] += [
        ('ebitda_smooth_new_AGENT', '(1.0 - rho_X) * ebitda_AGENT'
         ' + rho_X * ebitda_smooth_AGENT / pi'),
        ('om_bar_AGENT', '(debt_AGENT / pi) / (the_DE * ebitda_smooth_new_AGENT)'),
        ('z_om_bar_AGENT', '(np.log(om_bar_AGENT) - mu_om_ATYPE) / sig_om_ATYPE'),
        ('F_viol_AGENT', '0.5 * (1.0 + _ERF(z_om_bar_AGENT / np.sqrt(2.0)))'),
        ('f_viol_AGENT', 'np.exp(-0.5 * (z_om_bar_AGENT ** 2)) / (om_bar_AGENT * sig_om_ATYPE * np.sqrt(2 * np.pi))'),
        #
        ('xi_AGENT', 'kap_viol_ATYPE * F_viol_AGENT'), 
        ('d_xi_d_om_AGENT', 'kap_viol_ATYPE * f_viol_AGENT'),
        ('d_om_d_B_AGENT', 'om_bar_AGENT / debt_AGENT'),
        ('d_om_d_X_AGENT', '-om_bar_AGENT / ebitda_smooth_new_AGENT'),
        ('dX_dX_AGENT', 'rho_X / pi'),
        #
        ('Psi_flow_AGENT', '-(debt_AGENT / pi) * d_xi_d_om_AGENT * d_om_d_X_AGENT'),
        ]
    
    block.rules['expectations'] += [
        ('E_Psi_AGENT', '(Lam_1_AGENT_NEXT / Lam_0_AGENT) * Psi_AGENT_NEXT * dX_dX_AGENT_NEXT'),
        ]
    
    block.rules['optimality'] += [
        ('Psi_AGENT', 'Psi_flow_AGENT + E_Psi_AGENT - Psi_AGENT'),
        ]
    
    block.rules['analytical_steady'] += [
        ('ebitda_smooth_AGENT', '(1.0 - rho_X) * ebitda_AGENT'
         ' / (1.0 - rho_X / pi_bar)'),
        ('Psi_AGENT', 'Psi_flow_AGENT / (1.0 - bet_ATYPE * dX_dX_AGENT)'),
        ]
    
    return block

def debt_block() -> ModelBlock:
    """Imposes debt covenants"""
    
    block = ModelBlock()
    
    block.rules['intermediate'] += [
        ('Om_denom_INSTRUMENT_AGENT', '1.0 - bet_pi_AGENT * frac_INSTRUMENT_remaining'),
        ]
    
    block.rules['optimality'] += [
        ('Om_principal_INSTRUMENT_AGENT', 'E_Om_principal_INSTRUMENT_AGENT / Lam_0_AGENT - Om_principal_INSTRUMENT_AGENT'),
        ('Om_spread_INSTRUMENT_AGENT', 'E_Om_spread_INSTRUMENT_AGENT / Lam_0_AGENT - Om_spread_INSTRUMENT_AGENT'),
        ]
    
    block.rules['expectations'] += [
        ('E_Om_principal_INSTRUMENT_AGENT', 'Lam_1_nom_AGENT_NEXT * (marg_principal_flow_INSTRUMENT_AGENT_NEXT '
         '+ frac_debt_remaining * Om_principal_INSTRUMENT_AGENT_NEXT)'),
        ('E_Om_spread_INSTRUMENT_AGENT', 'Lam_1_nom_AGENT_NEXT * (marg_spread_flow_INSTRUMENT_AGENT_NEXT '
         '+ frac_debt_remaining * Om_spread_INSTRUMENT_AGENT_NEXT)'),
        ]
    
    block.rules['analytical_steady'] += [
        ('Om_principal_INSTRUMENT_AGENT', 'bet_pi_ATYPE * marg_principal_flow_INSTRUMENT_AGENT / Om_denom_INSTRUMENT_ATYPE'),
        ('Om_spread_INSTRUMENT_AGENT', 'bet_pi_ATYPE * marg_spread_flow_INSTRUMENT_AGENT / Om_denom_INSTRUMENT_ATYPE'),
        ]
    
    return block

def bank_block(
        holding_cost: bool = True, # TODO
        bank_capital_structure: bool = False, # TODO
        buffer_cost: bool = False,
        deposits: bool = False, # TODO
        is_main_type: bool = False,
        **kwargs,
        ) -> ModelBlock:
    """Banking sector"""
    
    block = ModelBlock()
    
    block.rules['intermediate'] += [
        # Security valuation
        ('N_security_BANK', 'N_security_bar_BTYPE'),
        ('share_afs_BANK', 'base_share_afs_BTYPE * (1.0 + x_afs_BANK)'),
        ('V_afs_BANK', 'pol_adjustment * share_afs_BANK * P_security_reg * N_security_BANK'),
        # TODO: would it be cleaner to track aoci as a state variable instead of using the book value?
        ('aoci_BANK', '(P_security_reg - P_security_book) * N_security_BANK'),
        ('mtm_share_BANK', 'pol_adjustment * share_afs_BANK'),
        ('securities_to_assets_BANK', 'P_security * N_security_bar_BTYPE / assets_BANK'),
        ('V_security_reg_BANK', '(mtm_share_BANK * P_security_reg + (1.0 - mtm_share_BANK) * P_security_book) * N_security_BANK'),
        #
        # Regulatory capital
        ('capital_BANK', 'assets_BANK- deposits_new_BANK'),
        ('capital_buffer_BANK', 'capital_BANK - required_capital_BANK'),
        ('required_capital_BANK', 'the_cap * loans_new_AGENT + the_other * other_assets_BTYPE'),
        #
        # General
        ('bet_pi_BANK', 'bet_BANK / pi_bar'),
        ('assets_BANK', 'loans_new_AGENT + V_security_reg_BANK + other_assets_BTYPE'),
        #
        # TODO: TEMPORARY
        # ('spread_L_AGENT', 'spread_L_bar'),
        #
        # Loan flows
        ('loan_issuance_AGENT', 'loans_new_AGENT - frac_debt_remaining * loans_AGENT / pi'),
        #
        # TODO: need to change if we tax the spread
        ('spread_payments_received_AGENT', 'spread_payments_AGENT'),
        #
        ('net_loan_flow_BANK', '(debt_base_payment * loans_AGENT + spread_payments_received_AGENT) / pi'
         ' - loan_issuance_AGENT'),
        ('net_security_flow_BANK', 'income_per_security * N_security_BANK'),
        #
        # TODO: temporary code
        ('D_BANK', 'net_loan_flow_BANK + net_security_flow_BANK - net_deposit_flow_BANK'),
        # ('marg_loan_cost_BANK', '0.0'),
        #
        # Loan valuation
        ('marg_principal_flow_loan_BANK', 'debt_base_payment + frac_debt_remaining * marg_loan_cost_BANK'),
        ('marg_spread_flow_loan_BANK', '1.0'),
        ('spread_L_implied_AGENT', '((spread_offset_ATYPE'
         ' + (1.0 - Om_principal_loan_BANK - marg_loan_cost_BANK'
         ' * (1.0 - the_cap)) / Om_spread_loan_BANK)'
         ' - spread_tax * spread_L_bar)'
         ' / (1.0 - spread_tax)'),    
        ]
    
    block.rules['optimality'] += [
        ('spread_L_AGENT', 'spread_L_AGENT - spread_L_implied_AGENT'),
        ]
    
    if holding_cost:
    # if True:
            
        block.rules['intermediate'] += [
            ('log_required_capital_no_aoci_BANK', 'np.log(required_capital_BANK - aoci_BANK)'),
            ('marg_loan_cost_BANK', 'loan_cost_mult_BTYPE'
             '* np.exp(loan_cost_exp * (log_required_capital_no_aoci_BANK - log_required_capital_no_aoci_bar_BTYPE))'),
            ('spread_L_implied_AGENT', 'spread_offset_ATYPE + (1.0 + Psi_loan_cost_S_BANK - Om_principal_debt_S) / Om_spread_debt_S'),
            ('deposits_new_BANK', '0.0'),
            ('net_deposit_flow_BANK', '0.0'),
            ]
        
        if is_main_type:
            block.rules['calibration'] += [
                ('loan_cost_mult_BANK', 'spread_L_AGENT - spread_L_bar'), # TODO: TEMPORARY
                # ('log_required_capital_no_aoci_bar_BANK', 'log_required_capital_no_aoci_BANK - log_required_capital_no_aoci_bar_BTYPE'),
                ]
        
        block.rules['optimality'] += [
            ('Psi_loan_cost_S_BANK', 'marg_loan_cost_BANK + E_Psi_loan_cost_S_BANK - Psi_loan_cost_S_BANK'),
        ]

        block.rules['expectations'] += [
            ('E_Psi_loan_cost_S_BANK', 'frac_debt_remaining * (Lam_1_S_NEXT / Lam_0_S) * Psi_loan_cost_S_BANK_NEXT'),
        ]
        
        block.rules['analytical_steady'] += [
            ('Psi_loan_cost_S_BANK', 'marg_loan_cost_BANK / (1.0 - frac_debt_remaining * bet_S)'),
            # ('spread_L_AGENT', 'spread_offset_ATYPE + (1.0 + Psi_loan_cost_S_BANK - Om_principal_debt_S) / Om_spread_debt_S'),
        ]
    
    # if deposits:
    else:
        
        block.rules['transition'] += [
            ('deposits_BANK', 'deposits_new_BANK'),
            ('deposit_payments_BANK', 'r_deposit_new * net_deposit_issuance_BANK + frac_deposits_remaining * deposit_payments_BANK / pi'),
            ]
        
        block.rules['analytical_steady'] += [
            ('deposits_BANK', 'deposits_new_BANK'),
            ('deposit_payments_BANK', '(r_deposit_new * net_deposit_issuance_BANK) / (1.0 - frac_deposits_remaining / pi_bar)'),
            ]
        
        block.rules['intermediate'] += [
            ('marg_principal_flow_deposits_BANK', 'frac_deposits_maturing'
             ' + frac_deposits_remaining * marg_loan_cost_BANK'),
            ('marg_spread_flow_deposits_BANK', '1.0'),
            #
            # TODO: temporary code
            ('r_deposit_new', 'r_new'),
            #
            ('net_deposit_issuance_BANK', 'deposits_new_BANK - frac_deposits_remaining * deposits_BANK / pi'),
            ('net_deposit_flow_BANK', '(deposit_payments_BANK + frac_deposits_maturing * deposits_BANK) / pi'
             ' - net_deposit_issuance_BANK'),
            ]
        
        block.add_block(
            debt_block(),
            rename={'AGENT' : 'BANK', 'ATYPE' : 'BTYPE', 'INSTRUMENT' : 'deposits'},
            )
        
        if buffer_cost:
        
            raise Exception
        
        else:
            
            block.rules['optimality'] += [
                ('marg_loan_cost_BANK', '1.0 - Om_principal_deposits_BANK'
                 ' - r_deposit_new * Om_spread_deposits_BANK - marg_loan_cost_BANK'),
                ]
            
            block.rules['intermediate'] += [
                ('deposits_new_BANK', '(1.0 - the_cap) * loans_new_AGENT + (1.0 - the_other) * other_assets_BTYPE + P_security_reg * N_security_BANK'),
                # TODO: TEMPORARY
                # ('net_deposit_flow_BANK', '0.0'),
                ]
        
    # else:
        
    #     block.rules['intermediate'] += [
    #         ('deposits_new_BANK', '0.0'),
    #         ('net_deposit_flow_BANK', '0.0'),
    #         ]
    
    return block

def bank_main_type_block() -> ModelBlock:
    """Calibrations for the main bank types"""
    
    block = ModelBlock()
    
    block.rules['calibration'] += [
        ('N_security_bar_BANK', 'P_security * N_security_bar_BANK / assets_BANK - asset_mult * securities_to_assets_bar'),
        ('base_share_afs_BANK', 'share_afs_BANK * P_security_reg * N_security_BANK / assets_BANK - afs_to_assets_target'),
        ]
    
    return block

def get_params(flags) -> dict:
    """Return baseline parameter dictionary."""
    
    # Size of regression step
    dx = 1e-4
    
    par = {
        # PREFERENCES
        'bet_S' : 0.995,
        'psi_S' : 0.1,
        'alp' : 0.33,
        'psi_ent' : 1.0,
        'bet_ent' : 0.99,
        #
        # TECHNOLOGY
        'delta' : 0.025,
        'lab_bar' : 1.0,
        'Z_bar' : 1.0,
        'pi_bar' : 1.005,
        'share_U' : 0.8,
        'K_share_U_bar' : 0.86,
        'inv_xi' : 0.25,
        'w_bar' : 1.0,
        #
        # GOVERNMENT
        'tau_corp' : 0.21,
        #
        # LT BONDS
        'security_maturity' : 4 * 4.0,
        #
        # DEBT
        'frac_debt_maturing' : 1.0,
        'spread_B_bar' : 0.025 / 4.0,
        'other_liabilities' : 0.0,
        'debt_to_liabilities' : 0.57,
        'spread_offset_U' : 0.0,
        'spread_offset_C' : 0.0,
        #
        # COVENANTS
        'rho_X' : 0.75,
        'the_DE' : 15.0,
        'log_sig_om_U' : -4.520e-01,
        'log_kap_viol_U' : -6.021e+00,
        'log_sig_om_C' : -4.520e-01,
        'log_kap_viol_C' : -6.021e+00,
        #
        # CASH BLOCK
        'cash_mult' : 0.009,
        'cash_exp' : 1.0,
        'cash_to_K_target_U' : 0.074,
        #
        # SMALL FIRM CALIBRATION
        'liabilities_to_assets_C' : 0.57,
        #
        # BANKS
        'bet_bank_C' : 0.99,
        'psi_bank_C' : 1.0,
        'pol_adjustment' : 1.0,
        'spread_L_bar' : 0.025 / 4.0,
        'other_assets_bank_C' : 0.0,
        'the_other' : 0.0,
        'securities_to_loans_bar' : 0.27356087620988284,
        'securities_to_assets_bar' : 0.2148,
        'base_share_afs_bank_C' : 1.0 / 3.5470552243744464,
        'asset_mult' : 1.0, # not sure why this is needed
        'afs_to_assets_target' : 0.1161 * 0.7,
        'spread_tax' : 0.0,
        'loan_cost_exp' : 1.0,
        'loan_cost_mult_bank_C' : 0.0025,
        'log_required_capital_no_aoci_bar_bank_C' : 0.0,
        #
        # DEPOSITS
        'frac_deposits_maturing' : 1.0,
        #
        # FOR REGRESSION
        'x_afs_bank_C' : 0.0,
        'x_afs_bank_C0' : -dx,
        'x_afs_bank_C1' : dx,
    }
    
    par.update({
        'leverage_bar_C' : par['debt_to_liabilities'] * par['liabilities_to_assets_C'],
        'cash_to_K_target_C' : 0.13 * par['liabilities_to_assets_C'],
        'N_security_bar_bank_C' : par['securities_to_loans_bar'] * 2.882e+00,
        })
    
    par['frac_debt_remaining'] = 1.0 - par['frac_debt_maturing']
    
    par.update({
        'frac_deposits_remaining' : 1.0 - par['frac_deposits_maturing'],
        'frac_loan_maturing' : par['frac_debt_maturing'],
        'frac_loan_remaining' : par['frac_debt_remaining'],
        })
    
    if flags['interest_on_cash']:
        par['cash_interest_share'] = 1.0
    else:
        par['cash_interest_share'] = 0.0
        
    if flags['buffer_cost']:
        par['the_cap'] = 0.08
    else:
        par['the_cap'] = 0.12
        
    for firm_type in ['U', 'C']:
        if flags[f'bonds_only_{firm_type}']:
            par[f'bond_weight_{firm_type}'] = 1.0
        else:
            par[f'bond_weight_{firm_type}'] = 0.0
    
    return par

def get_steady_guess(flags, params) -> dict:
    """Return initial steady state guesses."""
    
    guess = {
        'inv_rate' : 0.025,
        'log_K' : np.log(6.0),
        'R_new' : 1.005 / 0.995,
        'cash_new' : 2.0,
        'marg_loan_cost_bank' : 0.005,
        'spread_L_implied' : params['spread_L_bar'],
        'spread_L' : params['spread_L_bar'],
    }
    
    guess['debt_new'] = 0.3 * np.exp(guess['log_K'])
    guess['spread_payments'] = params['spread_L_bar'] * guess['debt_new']
        
    return guess

def create_model(
    flags: dict | None = None,
    params: dict | None = None,
    steady_guess: dict | None = None,
    label: str = "test",
    **kwargs,
) -> Model:
    
    # Merge with defaults
    default_flags = {
        'mark_to_market' : True,
        'fixed_wage' : True,
        'interest_on_cash' : True,
        'bonds_only_U' : True,
        'bonds_only_C' : False,
        'deposits' : True,
        'buffer_cost' : False,
        }
    
    flags = {**default_flags, **(flags or {})}    
    
    mod = Model(
        flags=flags,
        params=params,
        # steady_guess=get_default_steady_guess(),
        label=label,
        **kwargs,
    )
    
    mod.params.update(get_params(mod.flags))
    mod.steady_guess.update(get_steady_guess(mod.flags, mod.params))
    
    ######################################################################
    # BASIC SETUP
    ######################################################################
    
    mod.add_block(preference_block(preferences='crra'), 
                  rename={'AGENT' : 'S', 'ATYPE' : 'S'})
    
    if params:
        mod.params.update(params)

    if steady_guess:
        mod.steady_guess.update(steady_guess)
        

    mod.rules['intermediate'] += [
        ('Z', 'Z_bar + Z_til'),
        ('pi', 'pi_bar'),
        ('y_check', 'y - c_S - D - inv'),
        ('debt_base_payment', 'frac_debt_maturing + r_lag'),
    ]
    
    mod.add_exog('Z_til', pers=0.95, vol=0.1)
    
    ######################################################################
    # SAVER BLOCK
    ######################################################################

    mod.rules['intermediate'] += [
        ('c_S', 'wage_bill'),
        # ('frac_debt_remaining', '1.0 - frac_debt_maturing'),
        # ('frac_loan_maturing', 'frac_debt_maturing'),
        # ('frac_loan_remaining', 'frac_debt_remaining'),
        # ('debt_base_payment', 'frac_debt_maturing + r_lag'),
        ('marg_principal_flow_debt_S', 'debt_base_payment'),
        ('marg_spread_flow_debt_S', '1.0'),
    ]
    
    mod.add_block(
        debt_block(),
        rename={'AGENT' : 'S', 'INSTRUMENT' : 'debt', 'ATYPE' : 'S'},
        )
    
    # mod.add_block(
    #     debt_block(),
    #     rename={'AGENT' : 'S', 'INSTRUMENT' : 'deposits', 'ATYPE' : 'S'},
    #     )

    ######################################################################
    # ST BOND BLOCK
    ######################################################################

    mod.rules['expectations'] += [
        ('E_Lam_S', '(Lam_1_nom_S_NEXT / Lam_0_S)'),
    ]

    mod.rules['optimality'] += [
         ('R_new', 'E_Lam_S * R_new - 1.0'),
    ]
    
    mod.rules['transition'] += [
        ('R_lag', 'R_new'),
        ]
    
    mod.rules['intermediate'] += [
        ('r_lag', 'R_lag - 1.0'),
        ('r_new', 'R_new - 1.0'),
        ]
    
    mod.rules['analytical_steady'] += [
        ('R_new', 'pi_bar / bet_S'),
        ('R_lag', 'R_new'),
        ]
    
    ######################################################################
    # LT BOND BLOCK
    ######################################################################

    # spec.log_list += ['P_security']
    mod.add_exog('tv_security_shock', 0.95, 0.01)
    
    mod.rules['transition'] += [
        ('P_security_book', '(1.0 - (1.0 - nu_security) / pi) * P_security'
        ' + ((1.0 - nu_security) / pi) * P_security_book'),
        ]
    
    mod.rules['intermediate'] += [
        ('bet_pi_S', 'bet_S / pi_bar'),
        ('r_bar', 'pi_bar / bet_S - 1.0'),
        ('nu_security', '(1.0 + r_bar) / security_maturity - r_bar'),
        ('income_per_security', '(nu_security / pi - (1.0 - (1.0 - nu_security) / pi) * P_security)'),
        ]
    
    mod.rules['optimality'] += [
        ('P_security', 'E_P_security - P_security'),
        ]
    
    mod.rules['expectations'] += [
        ('E_P_security', '(Lam_1_nom_S_NEXT / Lam_0_S) * np.exp(tv_security_shock) * (nu_security + (1.0 - nu_security) * P_security_NEXT)'),
        ]
    
    mod.rules['analytical_steady'] += [
        ('P_security', 'bet_pi_S * (nu_security) / (1.0 - bet_pi_S * (1.0 - nu_security))'),
        ('P_security_book', 'P_security'),
        ]
    
    if mod.flags['mark_to_market']:
        mod.rules['intermediate'] += [
            ('P_security_reg', 'P_security'),
            ]
    else:
        mod.rules['intermediate'] += [
            ('P_security_reg', 'P_security_book'),
            ]

    ######################################################################
    # FIRMS
    ######################################################################
    
    base_firm_list = ['U', 'C']
    firm_list = base_firm_list.copy()
    
    firm_param_list = ['lab_bar', 'w_bar', 'cash_mult', 'other_liabilities']
    firm_steady_list = [
        'inv_rate', 'log_K', 'cash_new', 'debt_new', 'spread_payments',
        'marg_loan_cost_bank', 'spread_L_implied', 'spread_L',
        ]
    
    
    # Firm shares
    mod.rules['calibration'] += [
        ('share_U', 'K_share_U - K_share_U_bar'),
        ]
    
    mod.rules['intermediate'] += [
        ('share_C', '1.0 - share_U'),
        ('K_share_U', '(share_U * K_U) / K'),
        ('after_tax_corp', '1.0 - tau_corp'),
        ('r_cash_lag', '(R_lag - 1.0) * cash_interest_share'),
        ]
    
    for firm in firm_list:
        
        firm_type, is_main_type = get_agent_type(firm)
        
        for var in firm_param_list:
            mod.params[f'{var}_{firm}'] = mod.params[var]
            
        for var in firm_steady_list:
            mod.steady_guess[f'{var}_{firm}'] = mod.steady_guess[var]
        
        # Entrepreneur-specific parameters
        for var in ['psi', 'bet']:
            mod.rules['intermediate'] += [
                (f'{var}_{firm}', f'{var}_ent'),
                ]
            
        rename = {'AGENT' : firm, 'ATYPE' : firm_type, '_ERF' : 'jax.scipy.special.erf'}
        
        # Shortcuts to some firm-specific flags
        bonds_only = flags[f'bonds_only_{firm_type}']
        
        # Main firm block
        mod.add_block(
            firm_block(bonds_only=bonds_only, is_main_type=is_main_type, **mod.flags),
            rename=rename,
            )
        
        # Investment block
        mod.add_block(
            investment_block(),
            rename=rename,
            )
        
        # Entrepreneur preference block
        mod.add_block(
            preference_block(preferences='cara', consumption_var='D'),
            rename=rename,
            )
        
        # Cash block
        mod.add_block(
            cash_block(),
            rename=rename,
            )
        
        # Covenants
        mod.add_block(
            covenant_block(),
            rename=rename,
            )
        
        # Debt balances
        mod.add_block(
            debt_block(),
            rename={**rename, 'INSTRUMENT' : 'debt'},
            )
        
        if not bonds_only:
            
            bank = f'bank_{firm}'
            bank_type = f'bank_{firm_type}'
            
            rename_bank = {**rename, 'BANK' : bank, 'BTYPE' : bank_type}
            
            mod.add_block(
                bank_block(is_main_type=is_main_type, **mod.flags),
                rename=rename_bank,
                )
            
            mod.add_block(
                preference_block(preferences='cara', consumption_var='D'),
                rename={'AGENT' : bank, 'ATYPE' : bank_type},
                )
            
            mod.add_block(
                debt_block(),
                rename={'AGENT' : bank, 'ATYPE' : bank_type, 'INSTRUMENT' : 'loan'},
                )
            
            if is_main_type:
                
                mod.add_block(
                    bank_main_type_block(),
                    rename=rename_bank,
                    )
    
    ######################################################################
    # AGGREGATION
    ######################################################################

    for var in ['lab', 'K_new', 'K', 'inv', 'y', 'D', 'wage_bill']:
        rhs = ' + '.join([f'share_{firm} * {var}_{firm}' for firm in base_firm_list])
        mod.rules['intermediate'] += [(var, rhs)]
    
    ######################################################################
    # FINISH UP
    ######################################################################

    mod.finalize()

    return mod
