import jax
import jax.numpy as np


from typing import NamedTuple


class State(NamedTuple):
    Om_principal_debt_S: np.ndarray
    Om_spread_debt_S: np.ndarray
    R_new: np.ndarray
    P_security: np.ndarray
    inv_rate_U: np.ndarray
    debt_new_U: np.ndarray
    cash_new_U: np.ndarray
    Psi_U: np.ndarray
    Om_principal_debt_U: np.ndarray
    Om_spread_debt_U: np.ndarray
    inv_rate_C: np.ndarray
    debt_new_C: np.ndarray
    cash_new_C: np.ndarray
    Psi_C: np.ndarray
    Om_principal_debt_C: np.ndarray
    Om_spread_debt_C: np.ndarray
    spread_L_C: np.ndarray
    Psi_loan_cost_S_bank_C: np.ndarray
    Om_principal_loan_bank_C: np.ndarray
    Om_spread_loan_bank_C: np.ndarray
    R_lag: np.ndarray
    P_security_book: np.ndarray
    log_K_U: np.ndarray
    debt_U: np.ndarray
    spread_payments_U: np.ndarray
    cash_U: np.ndarray
    ebitda_smooth_U: np.ndarray
    log_K_C: np.ndarray
    debt_C: np.ndarray
    spread_payments_C: np.ndarray
    cash_C: np.ndarray
    ebitda_smooth_C: np.ndarray
    Z_til: np.ndarray
    tv_security_shock: np.ndarray
    E_Om_principal_debt_S: np.ndarray
    E_Om_spread_debt_S: np.ndarray
    E_Lam_S: np.ndarray
    E_P_security: np.ndarray
    E_Om_K_U: np.ndarray
    E_Om_cash_U: np.ndarray
    E_Psi_U: np.ndarray
    E_Om_principal_debt_U: np.ndarray
    E_Om_spread_debt_U: np.ndarray
    E_Om_K_C: np.ndarray
    E_Om_cash_C: np.ndarray
    E_Psi_C: np.ndarray
    E_Om_principal_debt_C: np.ndarray
    E_Om_spread_debt_C: np.ndarray
    E_Psi_loan_cost_S_bank_C: np.ndarray
    E_Om_principal_loan_bank_C: np.ndarray
    E_Om_spread_loan_bank_C: np.ndarray
    K_share_U_bar: np.ndarray
    N_security_bar_bank_C: np.ndarray
    PERS_Z_til: np.ndarray
    PERS_tv_security_shock: np.ndarray
    VOL_Z_til: np.ndarray
    VOL_tv_security_shock: np.ndarray
    Z_bar: np.ndarray
    afs_to_assets_target: np.ndarray
    alp: np.ndarray
    asset_mult: np.ndarray
    base_share_afs_bank_C: np.ndarray
    bet_S: np.ndarray
    bet_bank_C: np.ndarray
    bet_ent: np.ndarray
    bond_weight_C: np.ndarray
    bond_weight_U: np.ndarray
    cash_exp: np.ndarray
    cash_interest_share: np.ndarray
    cash_mult: np.ndarray
    cash_mult_C: np.ndarray
    cash_mult_U: np.ndarray
    cash_to_K_target_C: np.ndarray
    cash_to_K_target_U: np.ndarray
    debt_to_liabilities: np.ndarray
    delta: np.ndarray
    frac_debt_maturing: np.ndarray
    frac_debt_remaining: np.ndarray
    frac_deposits_maturing: np.ndarray
    frac_deposits_remaining: np.ndarray
    frac_loan_maturing: np.ndarray
    frac_loan_remaining: np.ndarray
    inv_xi: np.ndarray
    lab_bar: np.ndarray
    lab_bar_C: np.ndarray
    lab_bar_U: np.ndarray
    leverage_bar_C: np.ndarray
    liabilities_to_assets_C: np.ndarray
    loan_cost_exp: np.ndarray
    loan_cost_mult_bank_C: np.ndarray
    log_kap_viol_C: np.ndarray
    log_kap_viol_U: np.ndarray
    log_required_capital_no_aoci_bar_bank_C: np.ndarray
    log_sig_om_C: np.ndarray
    log_sig_om_U: np.ndarray
    other_assets_bank_C: np.ndarray
    other_liabilities: np.ndarray
    other_liabilities_C: np.ndarray
    other_liabilities_U: np.ndarray
    pi_bar: np.ndarray
    pol_adjustment: np.ndarray
    psi_S: np.ndarray
    psi_bank_C: np.ndarray
    psi_ent: np.ndarray
    rho_X: np.ndarray
    securities_to_assets_bar: np.ndarray
    securities_to_loans_bar: np.ndarray
    security_maturity: np.ndarray
    share_U: np.ndarray
    spread_B_bar: np.ndarray
    spread_L_bar: np.ndarray
    spread_offset_C: np.ndarray
    spread_offset_U: np.ndarray
    spread_tax: np.ndarray
    tau_corp: np.ndarray
    the_DE: np.ndarray
    the_cap: np.ndarray
    the_other: np.ndarray
    w_bar: np.ndarray
    w_bar_C: np.ndarray
    w_bar_U: np.ndarray
    x_afs_bank_C: np.ndarray
    x_afs_bank_C0: np.ndarray
    x_afs_bank_C1: np.ndarray
    pi: np.ndarray
    Z: np.ndarray
    r_lag: np.ndarray
    marg_spread_flow_debt_S: np.ndarray
    bet_pi_S: np.ndarray
    r_new: np.ndarray
    r_bar: np.ndarray
    P_security_reg: np.ndarray
    share_C: np.ndarray
    K_U: np.ndarray
    after_tax_corp: np.ndarray
    r_cash_lag: np.ndarray
    psi_U: np.ndarray
    bet_U: np.ndarray
    lab_U: np.ndarray
    spread_at_issuance_U: np.ndarray
    liabilities_new_U: np.ndarray
    loans_new_U: np.ndarray
    loans_U: np.ndarray
    sig_om_U: np.ndarray
    kap_viol_U: np.ndarray
    tv_inv_efficiency: np.ndarray
    inv_cost_1_U: np.ndarray
    du_cash_U: np.ndarray
    psi_C: np.ndarray
    bet_C: np.ndarray
    lab_C: np.ndarray
    K_C: np.ndarray
    spread_at_issuance_C: np.ndarray
    liabilities_new_C: np.ndarray
    loans_new_C: np.ndarray
    loans_C: np.ndarray
    sig_om_C: np.ndarray
    kap_viol_C: np.ndarray
    inv_cost_1_C: np.ndarray
    du_cash_C: np.ndarray
    N_security_bank_C: np.ndarray
    share_afs_bank_C: np.ndarray
    deposits_new_bank_C: np.ndarray
    bet_pi_bank_C: np.ndarray
    spread_payments_received_C: np.ndarray
    net_deposit_flow_bank_C: np.ndarray
    marg_spread_flow_loan_bank_C: np.ndarray
    spread_L_implied_C: np.ndarray
    debt_issuance_U: np.ndarray
    dX_dX_U: np.ndarray
    debt_issuance_C: np.ndarray
    dX_dX_C: np.ndarray
    debt_base_payment: np.ndarray
    Om_denom_debt_S: np.ndarray
    nu_security: np.ndarray
    inv_U: np.ndarray
    marg_spread_flow_debt_U: np.ndarray
    marg_spread_flow_debt_C: np.ndarray
    net_outflow_cash_U: np.ndarray
    net_outflow_cash_C: np.ndarray
    bet_pi_U: np.ndarray
    y_U: np.ndarray
    mu_om_U: np.ndarray
    d_Phi_inv_U: np.ndarray
    inv_cost_0_U: np.ndarray
    bet_pi_C: np.ndarray
    lab: np.ndarray
    y_C: np.ndarray
    inv_C: np.ndarray
    K: np.ndarray
    required_capital_bank_C: np.ndarray
    loan_issuance_C: np.ndarray
    mu_om_C: np.ndarray
    d_Phi_inv_C: np.ndarray
    inv_cost_0_C: np.ndarray
    aoci_bank_C: np.ndarray
    V_afs_bank_C: np.ndarray
    mtm_share_bank_C: np.ndarray
    Om_denom_loan_bank_C: np.ndarray
    spread_payments_new_U: np.ndarray
    spread_payments_new_C: np.ndarray
    marg_principal_flow_debt_S: np.ndarray
    income_per_security: np.ndarray
    K_new_U: np.ndarray
    Om_denom_debt_U: np.ndarray
    fk_U: np.ndarray
    flab_U: np.ndarray
    Q_U: np.ndarray
    Phi_inv_U: np.ndarray
    Om_denom_debt_C: np.ndarray
    fk_C: np.ndarray
    flab_C: np.ndarray
    y: np.ndarray
    K_new_C: np.ndarray
    inv: np.ndarray
    K_share_U: np.ndarray
    net_loan_flow_bank_C: np.ndarray
    Q_C: np.ndarray
    Phi_inv_C: np.ndarray
    log_required_capital_no_aoci_bank_C: np.ndarray
    V_security_reg_bank_C: np.ndarray
    net_security_flow_bank_C: np.ndarray
    wage_bill_U: np.ndarray
    Q_bar_U: np.ndarray
    wage_bill_C: np.ndarray
    K_new: np.ndarray
    Q_bar_C: np.ndarray
    marg_loan_cost_bank_C: np.ndarray
    assets_bank_C: np.ndarray
    D_bank_C: np.ndarray
    ebitda_U: np.ndarray
    Rk_Q_marg_U: np.ndarray
    ebitda_C: np.ndarray
    wage_bill: np.ndarray
    Rk_Q_marg_C: np.ndarray
    marg_principal_flow_loan_bank_C: np.ndarray
    securities_to_assets_bank_C: np.ndarray
    capital_bank_C: np.ndarray
    uc_bank_C: np.ndarray
    x_U: np.ndarray
    ebitda_smooth_new_U: np.ndarray
    x_C: np.ndarray
    ebitda_smooth_new_C: np.ndarray
    c_S: np.ndarray
    capital_buffer_bank_C: np.ndarray
    Lam_1_bank_C: np.ndarray
    Lam_0_bank_C: np.ndarray
    Rk_Q_avg_U: np.ndarray
    om_bar_U: np.ndarray
    Rk_Q_avg_C: np.ndarray
    om_bar_C: np.ndarray
    uc_S: np.ndarray
    Lam_1_nom_bank_C: np.ndarray
    z_om_bar_U: np.ndarray
    d_om_d_B_U: np.ndarray
    d_om_d_X_U: np.ndarray
    z_om_bar_C: np.ndarray
    d_om_d_B_C: np.ndarray
    d_om_d_X_C: np.ndarray
    Lam_1_S: np.ndarray
    Lam_0_S: np.ndarray
    F_viol_U: np.ndarray
    f_viol_U: np.ndarray
    F_viol_C: np.ndarray
    f_viol_C: np.ndarray
    Lam_1_nom_S: np.ndarray
    xi_U: np.ndarray
    d_xi_d_om_U: np.ndarray
    xi_C: np.ndarray
    d_xi_d_om_C: np.ndarray
    avg_principal_flow_debt_U: np.ndarray
    Psi_flow_U: np.ndarray
    avg_principal_flow_debt_C: np.ndarray
    Psi_flow_C: np.ndarray
    marg_principal_flow_debt_U: np.ndarray
    principal_payments_U: np.ndarray
    marg_principal_flow_debt_C: np.ndarray
    principal_payments_C: np.ndarray
    net_outflow_debt_U: np.ndarray
    net_outflow_debt_C: np.ndarray
    net_outflow_financial_U: np.ndarray
    net_outflow_financial_C: np.ndarray
    D_U: np.ndarray
    D_C: np.ndarray
    uc_U: np.ndarray
    uc_C: np.ndarray
    D: np.ndarray
    Lam_1_U: np.ndarray
    Lam_0_U: np.ndarray
    Lam_1_C: np.ndarray
    Lam_0_C: np.ndarray
    y_check: np.ndarray
    Lam_1_nom_U: np.ndarray
    Lam_1_nom_C: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)


def _upd(st: State, **kw):
    """Pure functional update: returns a new State with fields replaced."""
    return st._replace(**kw)


def array_to_state(x):
    """Initialize State from array x"""
    nan = np.nan * x[0]
    return State(
        Om_principal_debt_S=x[0],
        Om_spread_debt_S=x[1],
        R_new=x[2],
        P_security=x[3],
        inv_rate_U=x[4],
        debt_new_U=x[5],
        cash_new_U=x[6],
        Psi_U=x[7],
        Om_principal_debt_U=x[8],
        Om_spread_debt_U=x[9],
        inv_rate_C=x[10],
        debt_new_C=x[11],
        cash_new_C=x[12],
        Psi_C=x[13],
        Om_principal_debt_C=x[14],
        Om_spread_debt_C=x[15],
        spread_L_C=x[16],
        Psi_loan_cost_S_bank_C=x[17],
        Om_principal_loan_bank_C=x[18],
        Om_spread_loan_bank_C=x[19],
        R_lag=x[20],
        P_security_book=x[21],
        log_K_U=x[22],
        debt_U=x[23],
        spread_payments_U=x[24],
        cash_U=x[25],
        ebitda_smooth_U=x[26],
        log_K_C=x[27],
        debt_C=x[28],
        spread_payments_C=x[29],
        cash_C=x[30],
        ebitda_smooth_C=x[31],
        Z_til=x[32],
        tv_security_shock=x[33],
        E_Om_principal_debt_S=x[34],
        E_Om_spread_debt_S=x[35],
        E_Lam_S=x[36],
        E_P_security=x[37],
        E_Om_K_U=x[38],
        E_Om_cash_U=x[39],
        E_Psi_U=x[40],
        E_Om_principal_debt_U=x[41],
        E_Om_spread_debt_U=x[42],
        E_Om_K_C=x[43],
        E_Om_cash_C=x[44],
        E_Psi_C=x[45],
        E_Om_principal_debt_C=x[46],
        E_Om_spread_debt_C=x[47],
        E_Psi_loan_cost_S_bank_C=x[48],
        E_Om_principal_loan_bank_C=x[49],
        E_Om_spread_loan_bank_C=x[50],
        K_share_U_bar=x[51],
        N_security_bar_bank_C=x[52],
        PERS_Z_til=x[53],
        PERS_tv_security_shock=x[54],
        VOL_Z_til=x[55],
        VOL_tv_security_shock=x[56],
        Z_bar=x[57],
        afs_to_assets_target=x[58],
        alp=x[59],
        asset_mult=x[60],
        base_share_afs_bank_C=x[61],
        bet_S=x[62],
        bet_bank_C=x[63],
        bet_ent=x[64],
        bond_weight_C=x[65],
        bond_weight_U=x[66],
        cash_exp=x[67],
        cash_interest_share=x[68],
        cash_mult=x[69],
        cash_mult_C=x[70],
        cash_mult_U=x[71],
        cash_to_K_target_C=x[72],
        cash_to_K_target_U=x[73],
        debt_to_liabilities=x[74],
        delta=x[75],
        frac_debt_maturing=x[76],
        frac_debt_remaining=x[77],
        frac_deposits_maturing=x[78],
        frac_deposits_remaining=x[79],
        frac_loan_maturing=x[80],
        frac_loan_remaining=x[81],
        inv_xi=x[82],
        lab_bar=x[83],
        lab_bar_C=x[84],
        lab_bar_U=x[85],
        leverage_bar_C=x[86],
        liabilities_to_assets_C=x[87],
        loan_cost_exp=x[88],
        loan_cost_mult_bank_C=x[89],
        log_kap_viol_C=x[90],
        log_kap_viol_U=x[91],
        log_required_capital_no_aoci_bar_bank_C=x[92],
        log_sig_om_C=x[93],
        log_sig_om_U=x[94],
        other_assets_bank_C=x[95],
        other_liabilities=x[96],
        other_liabilities_C=x[97],
        other_liabilities_U=x[98],
        pi_bar=x[99],
        pol_adjustment=x[100],
        psi_S=x[101],
        psi_bank_C=x[102],
        psi_ent=x[103],
        rho_X=x[104],
        securities_to_assets_bar=x[105],
        securities_to_loans_bar=x[106],
        security_maturity=x[107],
        share_U=x[108],
        spread_B_bar=x[109],
        spread_L_bar=x[110],
        spread_offset_C=x[111],
        spread_offset_U=x[112],
        spread_tax=x[113],
        tau_corp=x[114],
        the_DE=x[115],
        the_cap=x[116],
        the_other=x[117],
        w_bar=x[118],
        w_bar_C=x[119],
        w_bar_U=x[120],
        x_afs_bank_C=x[121],
        x_afs_bank_C0=x[122],
        x_afs_bank_C1=x[123],
        pi=nan,
        Z=nan,
        r_lag=nan,
        marg_spread_flow_debt_S=nan,
        bet_pi_S=nan,
        r_new=nan,
        r_bar=nan,
        P_security_reg=nan,
        share_C=nan,
        K_U=nan,
        after_tax_corp=nan,
        r_cash_lag=nan,
        psi_U=nan,
        bet_U=nan,
        lab_U=nan,
        spread_at_issuance_U=nan,
        liabilities_new_U=nan,
        loans_new_U=nan,
        loans_U=nan,
        sig_om_U=nan,
        kap_viol_U=nan,
        tv_inv_efficiency=nan,
        inv_cost_1_U=nan,
        du_cash_U=nan,
        psi_C=nan,
        bet_C=nan,
        lab_C=nan,
        K_C=nan,
        spread_at_issuance_C=nan,
        liabilities_new_C=nan,
        loans_new_C=nan,
        loans_C=nan,
        sig_om_C=nan,
        kap_viol_C=nan,
        inv_cost_1_C=nan,
        du_cash_C=nan,
        N_security_bank_C=nan,
        share_afs_bank_C=nan,
        deposits_new_bank_C=nan,
        bet_pi_bank_C=nan,
        spread_payments_received_C=nan,
        net_deposit_flow_bank_C=nan,
        marg_spread_flow_loan_bank_C=nan,
        spread_L_implied_C=nan,
        debt_issuance_U=nan,
        dX_dX_U=nan,
        debt_issuance_C=nan,
        dX_dX_C=nan,
        debt_base_payment=nan,
        Om_denom_debt_S=nan,
        nu_security=nan,
        inv_U=nan,
        marg_spread_flow_debt_U=nan,
        marg_spread_flow_debt_C=nan,
        net_outflow_cash_U=nan,
        net_outflow_cash_C=nan,
        bet_pi_U=nan,
        y_U=nan,
        mu_om_U=nan,
        d_Phi_inv_U=nan,
        inv_cost_0_U=nan,
        bet_pi_C=nan,
        lab=nan,
        y_C=nan,
        inv_C=nan,
        K=nan,
        required_capital_bank_C=nan,
        loan_issuance_C=nan,
        mu_om_C=nan,
        d_Phi_inv_C=nan,
        inv_cost_0_C=nan,
        aoci_bank_C=nan,
        V_afs_bank_C=nan,
        mtm_share_bank_C=nan,
        Om_denom_loan_bank_C=nan,
        spread_payments_new_U=nan,
        spread_payments_new_C=nan,
        marg_principal_flow_debt_S=nan,
        income_per_security=nan,
        K_new_U=nan,
        Om_denom_debt_U=nan,
        fk_U=nan,
        flab_U=nan,
        Q_U=nan,
        Phi_inv_U=nan,
        Om_denom_debt_C=nan,
        fk_C=nan,
        flab_C=nan,
        y=nan,
        K_new_C=nan,
        inv=nan,
        K_share_U=nan,
        net_loan_flow_bank_C=nan,
        Q_C=nan,
        Phi_inv_C=nan,
        log_required_capital_no_aoci_bank_C=nan,
        V_security_reg_bank_C=nan,
        net_security_flow_bank_C=nan,
        wage_bill_U=nan,
        Q_bar_U=nan,
        wage_bill_C=nan,
        K_new=nan,
        Q_bar_C=nan,
        marg_loan_cost_bank_C=nan,
        assets_bank_C=nan,
        D_bank_C=nan,
        ebitda_U=nan,
        Rk_Q_marg_U=nan,
        ebitda_C=nan,
        wage_bill=nan,
        Rk_Q_marg_C=nan,
        marg_principal_flow_loan_bank_C=nan,
        securities_to_assets_bank_C=nan,
        capital_bank_C=nan,
        uc_bank_C=nan,
        x_U=nan,
        ebitda_smooth_new_U=nan,
        x_C=nan,
        ebitda_smooth_new_C=nan,
        c_S=nan,
        capital_buffer_bank_C=nan,
        Lam_1_bank_C=nan,
        Lam_0_bank_C=nan,
        Rk_Q_avg_U=nan,
        om_bar_U=nan,
        Rk_Q_avg_C=nan,
        om_bar_C=nan,
        uc_S=nan,
        Lam_1_nom_bank_C=nan,
        z_om_bar_U=nan,
        d_om_d_B_U=nan,
        d_om_d_X_U=nan,
        z_om_bar_C=nan,
        d_om_d_B_C=nan,
        d_om_d_X_C=nan,
        Lam_1_S=nan,
        Lam_0_S=nan,
        F_viol_U=nan,
        f_viol_U=nan,
        F_viol_C=nan,
        f_viol_C=nan,
        Lam_1_nom_S=nan,
        xi_U=nan,
        d_xi_d_om_U=nan,
        xi_C=nan,
        d_xi_d_om_C=nan,
        avg_principal_flow_debt_U=nan,
        Psi_flow_U=nan,
        avg_principal_flow_debt_C=nan,
        Psi_flow_C=nan,
        marg_principal_flow_debt_U=nan,
        principal_payments_U=nan,
        marg_principal_flow_debt_C=nan,
        principal_payments_C=nan,
        net_outflow_debt_U=nan,
        net_outflow_debt_C=nan,
        net_outflow_financial_U=nan,
        net_outflow_financial_C=nan,
        D_U=nan,
        D_C=nan,
        uc_U=nan,
        uc_C=nan,
        D=nan,
        Lam_1_U=nan,
        Lam_0_U=nan,
        Lam_1_C=nan,
        Lam_0_C=nan,
        y_check=nan,
        Lam_1_nom_U=nan,
        Lam_1_nom_C=nan,
    )


def transition_inner(st):
    return np.array(
        (
            st.R_new,
            (1.0 - (1.0 - st.nu_security) / st.pi) * st.P_security
            + ((1.0 - st.nu_security) / st.pi) * st.P_security_book,
            np.log(st.K_new_U),
            st.debt_new_U,
            st.spread_payments_new_U,
            st.cash_new_U,
            st.ebitda_smooth_new_U,
            np.log(st.K_new_C),
            st.debt_new_C,
            st.spread_payments_new_C,
            st.cash_new_C,
            st.ebitda_smooth_new_C,
        )
    )


def expectations_inner(st, st_new):
    return np.array(
        (
            st_new.Lam_1_nom_S
            * (
                st_new.marg_principal_flow_debt_S
                + st.frac_debt_remaining * st_new.Om_principal_debt_S
            ),
            st_new.Lam_1_nom_S
            * (
                st_new.marg_spread_flow_debt_S
                + st.frac_debt_remaining * st_new.Om_spread_debt_S
            ),
            (st_new.Lam_1_nom_S / st.Lam_0_S),
            (st_new.Lam_1_nom_S / st.Lam_0_S)
            * np.exp(st.tv_security_shock)
            * (st.nu_security + (1.0 - st.nu_security) * st_new.P_security),
            (st_new.Lam_1_U / st.Lam_0_U)
            * (st_new.Rk_Q_marg_U + (1.0 - st.rho_X) * st_new.Psi_U * st_new.fk_U),
            (st_new.Lam_1_nom_U / st.Lam_0_U) * (1.0 + st_new.r_cash_lag),
            (st_new.Lam_1_U / st.Lam_0_U) * st_new.Psi_U * st_new.dX_dX_U,
            st_new.Lam_1_nom_U
            * (
                st_new.marg_principal_flow_debt_U
                + st.frac_debt_remaining * st_new.Om_principal_debt_U
            ),
            st_new.Lam_1_nom_U
            * (
                st_new.marg_spread_flow_debt_U
                + st.frac_debt_remaining * st_new.Om_spread_debt_U
            ),
            (st_new.Lam_1_C / st.Lam_0_C)
            * (st_new.Rk_Q_marg_C + (1.0 - st.rho_X) * st_new.Psi_C * st_new.fk_C),
            (st_new.Lam_1_nom_C / st.Lam_0_C) * (1.0 + st_new.r_cash_lag),
            (st_new.Lam_1_C / st.Lam_0_C) * st_new.Psi_C * st_new.dX_dX_C,
            st_new.Lam_1_nom_C
            * (
                st_new.marg_principal_flow_debt_C
                + st.frac_debt_remaining * st_new.Om_principal_debt_C
            ),
            st_new.Lam_1_nom_C
            * (
                st_new.marg_spread_flow_debt_C
                + st.frac_debt_remaining * st_new.Om_spread_debt_C
            ),
            st.frac_debt_remaining
            * (st_new.Lam_1_S / st.Lam_0_S)
            * st_new.Psi_loan_cost_S_bank_C,
            st_new.Lam_1_nom_bank_C
            * (
                st_new.marg_principal_flow_loan_bank_C
                + st.frac_debt_remaining * st_new.Om_principal_loan_bank_C
            ),
            st_new.Lam_1_nom_bank_C
            * (
                st_new.marg_spread_flow_loan_bank_C
                + st.frac_debt_remaining * st_new.Om_spread_loan_bank_C
            ),
        )
    )


def optimality_inner(st):
    return np.array(
        (
            st.E_Om_principal_debt_S / st.Lam_0_S - st.Om_principal_debt_S,
            st.E_Om_spread_debt_S / st.Lam_0_S - st.Om_spread_debt_S,
            st.E_Lam_S * st.R_new - 1.0,
            st.E_P_security - st.P_security,
            st.E_Om_K_U - st.Q_U,
            1.0
            - st.Om_principal_debt_U
            - st.Om_spread_debt_U * st.spread_at_issuance_U,
            (1.0 - st.du_cash_U) - st.E_Om_cash_U,
            st.Psi_flow_U + st.E_Psi_U - st.Psi_U,
            st.E_Om_principal_debt_U / st.Lam_0_U - st.Om_principal_debt_U,
            st.E_Om_spread_debt_U / st.Lam_0_U - st.Om_spread_debt_U,
            st.E_Om_K_C - st.Q_C,
            1.0
            - st.Om_principal_debt_C
            - st.Om_spread_debt_C * st.spread_at_issuance_C,
            (1.0 - st.du_cash_C) - st.E_Om_cash_C,
            st.Psi_flow_C + st.E_Psi_C - st.Psi_C,
            st.E_Om_principal_debt_C / st.Lam_0_C - st.Om_principal_debt_C,
            st.E_Om_spread_debt_C / st.Lam_0_C - st.Om_spread_debt_C,
            st.spread_L_C - st.spread_L_implied_C,
            st.marg_loan_cost_bank_C
            + st.E_Psi_loan_cost_S_bank_C
            - st.Psi_loan_cost_S_bank_C,
            st.E_Om_principal_loan_bank_C / st.Lam_0_bank_C
            - st.Om_principal_loan_bank_C,
            st.E_Om_spread_loan_bank_C / st.Lam_0_bank_C - st.Om_spread_loan_bank_C,
        )
    )


def intermediate_variables(st):
    pi = st.pi_bar
    Z = st.Z_bar + st.Z_til
    r_lag = st.R_lag - 1.0
    marg_spread_flow_debt_S = 1.0
    bet_pi_S = st.bet_S / st.pi_bar
    r_new = st.R_new - 1.0
    r_bar = st.pi_bar / st.bet_S - 1.0
    P_security_reg = st.P_security
    share_C = 1.0 - st.share_U
    K_U = np.exp(st.log_K_U)
    after_tax_corp = 1.0 - st.tau_corp
    r_cash_lag = (st.R_lag - 1.0) * st.cash_interest_share
    psi_U = st.psi_ent
    bet_U = st.bet_ent
    lab_U = st.lab_bar_U
    spread_at_issuance_U = st.spread_B_bar
    liabilities_new_U = st.debt_new_U + st.other_liabilities_U
    loans_new_U = (1.0 - st.bond_weight_U) * st.debt_new_U
    loans_U = (1.0 - st.bond_weight_U) * st.debt_U
    sig_om_U = np.exp(st.log_sig_om_U)
    kap_viol_U = np.exp(st.log_kap_viol_U)
    tv_inv_efficiency = 1.0
    inv_cost_1_U = st.delta**st.inv_xi
    du_cash_U = st.cash_mult_U * (st.cash_new_U ** (-st.cash_exp))
    psi_C = st.psi_ent
    bet_C = st.bet_ent
    lab_C = st.lab_bar_C
    K_C = np.exp(st.log_K_C)
    spread_at_issuance_C = (
        st.bond_weight_C * st.spread_B_bar + (1.0 - st.bond_weight_C) * st.spread_L_C
    )
    liabilities_new_C = st.debt_new_C + st.other_liabilities_C
    loans_new_C = (1.0 - st.bond_weight_C) * st.debt_new_C
    loans_C = (1.0 - st.bond_weight_C) * st.debt_C
    sig_om_C = np.exp(st.log_sig_om_C)
    kap_viol_C = np.exp(st.log_kap_viol_C)
    inv_cost_1_C = st.delta**st.inv_xi
    du_cash_C = st.cash_mult_C * (st.cash_new_C ** (-st.cash_exp))
    N_security_bank_C = st.N_security_bar_bank_C
    share_afs_bank_C = st.base_share_afs_bank_C * (1.0 + st.x_afs_bank_C)
    deposits_new_bank_C = 0.0
    bet_pi_bank_C = st.bet_bank_C / st.pi_bar
    spread_payments_received_C = st.spread_payments_C
    net_deposit_flow_bank_C = 0.0
    marg_spread_flow_loan_bank_C = 1.0
    spread_L_implied_C = (
        st.spread_offset_C
        + (1.0 + st.Psi_loan_cost_S_bank_C - st.Om_principal_debt_S)
        / st.Om_spread_debt_S
    )
    debt_issuance_U = st.debt_new_U - st.frac_debt_remaining * st.debt_U / pi
    dX_dX_U = st.rho_X / pi
    debt_issuance_C = st.debt_new_C - st.frac_debt_remaining * st.debt_C / pi
    dX_dX_C = st.rho_X / pi
    debt_base_payment = st.frac_debt_maturing + r_lag
    Om_denom_debt_S = 1.0 - bet_pi_S * st.frac_debt_remaining
    nu_security = (1.0 + r_bar) / st.security_maturity - r_bar
    inv_U = st.inv_rate_U * K_U
    marg_spread_flow_debt_U = after_tax_corp
    marg_spread_flow_debt_C = after_tax_corp
    net_outflow_cash_U = st.cash_new_U - (1.0 + r_cash_lag) * st.cash_U / pi
    net_outflow_cash_C = st.cash_new_C - (1.0 + r_cash_lag) * st.cash_C / pi
    bet_pi_U = bet_U / st.pi_bar
    y_U = Z * (K_U**st.alp) * (lab_U ** (1.0 - st.alp))
    mu_om_U = -0.5 * (sig_om_U**2)
    d_Phi_inv_U = inv_cost_1_U * tv_inv_efficiency * pow(st.inv_rate_U, -st.inv_xi)
    inv_cost_0_U = st.delta - inv_cost_1_U * (st.delta ** (1.0 - st.inv_xi)) / (
        1.0 - st.inv_xi
    )
    bet_pi_C = bet_C / st.pi_bar
    lab = st.share_U * lab_U + share_C * lab_C
    y_C = Z * (K_C**st.alp) * (lab_C ** (1.0 - st.alp))
    inv_C = st.inv_rate_C * K_C
    K = st.share_U * K_U + share_C * K_C
    required_capital_bank_C = (
        st.the_cap * loans_new_C + st.the_other * st.other_assets_bank_C
    )
    loan_issuance_C = loans_new_C - st.frac_debt_remaining * loans_C / pi
    mu_om_C = -0.5 * (sig_om_C**2)
    d_Phi_inv_C = inv_cost_1_C * tv_inv_efficiency * pow(st.inv_rate_C, -st.inv_xi)
    inv_cost_0_C = st.delta - inv_cost_1_C * (st.delta ** (1.0 - st.inv_xi)) / (
        1.0 - st.inv_xi
    )
    aoci_bank_C = (P_security_reg - st.P_security_book) * N_security_bank_C
    V_afs_bank_C = (
        st.pol_adjustment * share_afs_bank_C * P_security_reg * N_security_bank_C
    )
    mtm_share_bank_C = st.pol_adjustment * share_afs_bank_C
    Om_denom_loan_bank_C = 1.0 - bet_pi_bank_C * st.frac_loan_remaining
    spread_payments_new_U = (
        debt_issuance_U * spread_at_issuance_U
        + st.frac_debt_remaining * st.spread_payments_U / pi
    )
    spread_payments_new_C = (
        debt_issuance_C * spread_at_issuance_C
        + st.frac_debt_remaining * st.spread_payments_C / pi
    )
    marg_principal_flow_debt_S = debt_base_payment
    income_per_security = (
        nu_security / pi - (1.0 - (1.0 - nu_security) / pi) * st.P_security
    )
    K_new_U = inv_U + (1.0 - st.delta) * K_U
    Om_denom_debt_U = 1.0 - bet_pi_U * st.frac_debt_remaining
    fk_U = st.alp * y_U / K_U
    flab_U = (1.0 - st.alp) * y_U / lab_U
    Q_U = 1.0 / d_Phi_inv_U
    Phi_inv_U = tv_inv_efficiency * (
        inv_cost_0_U
        + inv_cost_1_U * pow(st.inv_rate_U, 1.0 - st.inv_xi) / (1.0 - st.inv_xi)
    )
    Om_denom_debt_C = 1.0 - bet_pi_C * st.frac_debt_remaining
    fk_C = st.alp * y_C / K_C
    flab_C = (1.0 - st.alp) * y_C / lab_C
    y = st.share_U * y_U + share_C * y_C
    K_new_C = inv_C + (1.0 - st.delta) * K_C
    inv = st.share_U * inv_U + share_C * inv_C
    K_share_U = (st.share_U * K_U) / K
    net_loan_flow_bank_C = (
        debt_base_payment * loans_C + spread_payments_received_C
    ) / pi - loan_issuance_C
    Q_C = 1.0 / d_Phi_inv_C
    Phi_inv_C = tv_inv_efficiency * (
        inv_cost_0_C
        + inv_cost_1_C * pow(st.inv_rate_C, 1.0 - st.inv_xi) / (1.0 - st.inv_xi)
    )
    log_required_capital_no_aoci_bank_C = np.log(required_capital_bank_C - aoci_bank_C)
    V_security_reg_bank_C = (
        mtm_share_bank_C * P_security_reg
        + (1.0 - mtm_share_bank_C) * st.P_security_book
    ) * N_security_bank_C
    net_security_flow_bank_C = income_per_security * N_security_bank_C
    wage_bill_U = flab_U * lab_U
    Q_bar_U = Q_U + (Q_U * Phi_inv_U - st.inv_rate_U) / (1.0 - st.delta)
    wage_bill_C = flab_C * lab_C
    K_new = st.share_U * K_new_U + share_C * K_new_C
    Q_bar_C = Q_C + (Q_C * Phi_inv_C - st.inv_rate_C) / (1.0 - st.delta)
    marg_loan_cost_bank_C = st.loan_cost_mult_bank_C * np.exp(
        st.loan_cost_exp
        * (
            log_required_capital_no_aoci_bank_C
            - st.log_required_capital_no_aoci_bar_bank_C
        )
    )
    assets_bank_C = loans_new_C + V_security_reg_bank_C + st.other_assets_bank_C
    D_bank_C = net_loan_flow_bank_C + net_security_flow_bank_C - net_deposit_flow_bank_C
    ebitda_U = y_U - wage_bill_U
    Rk_Q_marg_U = after_tax_corp * fk_U + (1.0 - after_tax_corp * st.delta) * Q_bar_U
    ebitda_C = y_C - wage_bill_C
    wage_bill = st.share_U * wage_bill_U + share_C * wage_bill_C
    Rk_Q_marg_C = after_tax_corp * fk_C + (1.0 - after_tax_corp * st.delta) * Q_bar_C
    marg_principal_flow_loan_bank_C = (
        debt_base_payment + st.frac_debt_remaining * marg_loan_cost_bank_C
    )
    securities_to_assets_bank_C = (
        st.P_security * st.N_security_bar_bank_C / assets_bank_C
    )
    capital_bank_C = assets_bank_C - deposits_new_bank_C
    uc_bank_C = np.exp(-st.psi_bank_C * D_bank_C)
    x_U = ebitda_U / K_U
    ebitda_smooth_new_U = (
        1.0 - st.rho_X
    ) * ebitda_U + st.rho_X * st.ebitda_smooth_U / pi
    x_C = ebitda_C / K_C
    ebitda_smooth_new_C = (
        1.0 - st.rho_X
    ) * ebitda_C + st.rho_X * st.ebitda_smooth_C / pi
    c_S = wage_bill
    capital_buffer_bank_C = capital_bank_C - required_capital_bank_C
    Lam_1_bank_C = uc_bank_C
    Lam_0_bank_C = uc_bank_C / st.bet_bank_C
    Rk_Q_avg_U = after_tax_corp * x_U + (1.0 - after_tax_corp * st.delta) * Q_bar_U
    om_bar_U = (st.debt_U / pi) / (st.the_DE * ebitda_smooth_new_U)
    Rk_Q_avg_C = after_tax_corp * x_C + (1.0 - after_tax_corp * st.delta) * Q_bar_C
    om_bar_C = (st.debt_C / pi) / (st.the_DE * ebitda_smooth_new_C)
    uc_S = c_S ** (-st.psi_S)
    Lam_1_nom_bank_C = Lam_1_bank_C / pi
    z_om_bar_U = (np.log(om_bar_U) - mu_om_U) / sig_om_U
    d_om_d_B_U = om_bar_U / st.debt_U
    d_om_d_X_U = -om_bar_U / ebitda_smooth_new_U
    z_om_bar_C = (np.log(om_bar_C) - mu_om_C) / sig_om_C
    d_om_d_B_C = om_bar_C / st.debt_C
    d_om_d_X_C = -om_bar_C / ebitda_smooth_new_C
    Lam_1_S = uc_S
    Lam_0_S = uc_S / st.bet_S
    F_viol_U = 0.5 * (1.0 + jax.scipy.special.erf(z_om_bar_U / np.sqrt(2.0)))
    f_viol_U = np.exp(-0.5 * (z_om_bar_U**2)) / (
        om_bar_U * sig_om_U * np.sqrt(2 * np.pi)
    )
    F_viol_C = 0.5 * (1.0 + jax.scipy.special.erf(z_om_bar_C / np.sqrt(2.0)))
    f_viol_C = np.exp(-0.5 * (z_om_bar_C**2)) / (
        om_bar_C * sig_om_C * np.sqrt(2 * np.pi)
    )
    Lam_1_nom_S = Lam_1_S / pi
    xi_U = kap_viol_U * F_viol_U
    d_xi_d_om_U = kap_viol_U * f_viol_U
    xi_C = kap_viol_C * F_viol_C
    d_xi_d_om_C = kap_viol_C * f_viol_C
    avg_principal_flow_debt_U = st.frac_debt_maturing + after_tax_corp * r_lag + xi_U
    Psi_flow_U = -(st.debt_U / pi) * d_xi_d_om_U * d_om_d_X_U
    avg_principal_flow_debt_C = st.frac_debt_maturing + after_tax_corp * r_lag + xi_C
    Psi_flow_C = -(st.debt_C / pi) * d_xi_d_om_C * d_om_d_X_C
    marg_principal_flow_debt_U = (
        avg_principal_flow_debt_U + d_xi_d_om_U * d_om_d_B_U * st.debt_U
    )
    principal_payments_U = avg_principal_flow_debt_U * st.debt_U
    marg_principal_flow_debt_C = (
        avg_principal_flow_debt_C + d_xi_d_om_C * d_om_d_B_C * st.debt_C
    )
    principal_payments_C = avg_principal_flow_debt_C * st.debt_C
    net_outflow_debt_U = (
        principal_payments_U + st.spread_payments_U
    ) / pi - debt_issuance_U
    net_outflow_debt_C = (
        principal_payments_C + st.spread_payments_C
    ) / pi - debt_issuance_C
    net_outflow_financial_U = net_outflow_debt_U + net_outflow_cash_U
    net_outflow_financial_C = net_outflow_debt_C + net_outflow_cash_C
    D_U = Rk_Q_avg_U * K_U - net_outflow_financial_U - K_new_U * Q_U
    D_C = Rk_Q_avg_C * K_C - net_outflow_financial_C - K_new_C * Q_C
    uc_U = np.exp(-psi_U * D_U)
    uc_C = np.exp(-psi_C * D_C)
    D = st.share_U * D_U + share_C * D_C
    Lam_1_U = uc_U
    Lam_0_U = uc_U / bet_U
    Lam_1_C = uc_C
    Lam_0_C = uc_C / bet_C
    y_check = y - c_S - D - inv
    Lam_1_nom_U = Lam_1_U / pi
    Lam_1_nom_C = Lam_1_C / pi
    return _upd(
        st,
        pi=pi,
        Z=Z,
        r_lag=r_lag,
        marg_spread_flow_debt_S=marg_spread_flow_debt_S,
        bet_pi_S=bet_pi_S,
        r_new=r_new,
        r_bar=r_bar,
        P_security_reg=P_security_reg,
        share_C=share_C,
        K_U=K_U,
        after_tax_corp=after_tax_corp,
        r_cash_lag=r_cash_lag,
        psi_U=psi_U,
        bet_U=bet_U,
        lab_U=lab_U,
        spread_at_issuance_U=spread_at_issuance_U,
        liabilities_new_U=liabilities_new_U,
        loans_new_U=loans_new_U,
        loans_U=loans_U,
        sig_om_U=sig_om_U,
        kap_viol_U=kap_viol_U,
        tv_inv_efficiency=tv_inv_efficiency,
        inv_cost_1_U=inv_cost_1_U,
        du_cash_U=du_cash_U,
        psi_C=psi_C,
        bet_C=bet_C,
        lab_C=lab_C,
        K_C=K_C,
        spread_at_issuance_C=spread_at_issuance_C,
        liabilities_new_C=liabilities_new_C,
        loans_new_C=loans_new_C,
        loans_C=loans_C,
        sig_om_C=sig_om_C,
        kap_viol_C=kap_viol_C,
        inv_cost_1_C=inv_cost_1_C,
        du_cash_C=du_cash_C,
        N_security_bank_C=N_security_bank_C,
        share_afs_bank_C=share_afs_bank_C,
        deposits_new_bank_C=deposits_new_bank_C,
        bet_pi_bank_C=bet_pi_bank_C,
        spread_payments_received_C=spread_payments_received_C,
        net_deposit_flow_bank_C=net_deposit_flow_bank_C,
        marg_spread_flow_loan_bank_C=marg_spread_flow_loan_bank_C,
        spread_L_implied_C=spread_L_implied_C,
        debt_issuance_U=debt_issuance_U,
        dX_dX_U=dX_dX_U,
        debt_issuance_C=debt_issuance_C,
        dX_dX_C=dX_dX_C,
        debt_base_payment=debt_base_payment,
        Om_denom_debt_S=Om_denom_debt_S,
        nu_security=nu_security,
        inv_U=inv_U,
        marg_spread_flow_debt_U=marg_spread_flow_debt_U,
        marg_spread_flow_debt_C=marg_spread_flow_debt_C,
        net_outflow_cash_U=net_outflow_cash_U,
        net_outflow_cash_C=net_outflow_cash_C,
        bet_pi_U=bet_pi_U,
        y_U=y_U,
        mu_om_U=mu_om_U,
        d_Phi_inv_U=d_Phi_inv_U,
        inv_cost_0_U=inv_cost_0_U,
        bet_pi_C=bet_pi_C,
        lab=lab,
        y_C=y_C,
        inv_C=inv_C,
        K=K,
        required_capital_bank_C=required_capital_bank_C,
        loan_issuance_C=loan_issuance_C,
        mu_om_C=mu_om_C,
        d_Phi_inv_C=d_Phi_inv_C,
        inv_cost_0_C=inv_cost_0_C,
        aoci_bank_C=aoci_bank_C,
        V_afs_bank_C=V_afs_bank_C,
        mtm_share_bank_C=mtm_share_bank_C,
        Om_denom_loan_bank_C=Om_denom_loan_bank_C,
        spread_payments_new_U=spread_payments_new_U,
        spread_payments_new_C=spread_payments_new_C,
        marg_principal_flow_debt_S=marg_principal_flow_debt_S,
        income_per_security=income_per_security,
        K_new_U=K_new_U,
        Om_denom_debt_U=Om_denom_debt_U,
        fk_U=fk_U,
        flab_U=flab_U,
        Q_U=Q_U,
        Phi_inv_U=Phi_inv_U,
        Om_denom_debt_C=Om_denom_debt_C,
        fk_C=fk_C,
        flab_C=flab_C,
        y=y,
        K_new_C=K_new_C,
        inv=inv,
        K_share_U=K_share_U,
        net_loan_flow_bank_C=net_loan_flow_bank_C,
        Q_C=Q_C,
        Phi_inv_C=Phi_inv_C,
        log_required_capital_no_aoci_bank_C=log_required_capital_no_aoci_bank_C,
        V_security_reg_bank_C=V_security_reg_bank_C,
        net_security_flow_bank_C=net_security_flow_bank_C,
        wage_bill_U=wage_bill_U,
        Q_bar_U=Q_bar_U,
        wage_bill_C=wage_bill_C,
        K_new=K_new,
        Q_bar_C=Q_bar_C,
        marg_loan_cost_bank_C=marg_loan_cost_bank_C,
        assets_bank_C=assets_bank_C,
        D_bank_C=D_bank_C,
        ebitda_U=ebitda_U,
        Rk_Q_marg_U=Rk_Q_marg_U,
        ebitda_C=ebitda_C,
        wage_bill=wage_bill,
        Rk_Q_marg_C=Rk_Q_marg_C,
        marg_principal_flow_loan_bank_C=marg_principal_flow_loan_bank_C,
        securities_to_assets_bank_C=securities_to_assets_bank_C,
        capital_bank_C=capital_bank_C,
        uc_bank_C=uc_bank_C,
        x_U=x_U,
        ebitda_smooth_new_U=ebitda_smooth_new_U,
        x_C=x_C,
        ebitda_smooth_new_C=ebitda_smooth_new_C,
        c_S=c_S,
        capital_buffer_bank_C=capital_buffer_bank_C,
        Lam_1_bank_C=Lam_1_bank_C,
        Lam_0_bank_C=Lam_0_bank_C,
        Rk_Q_avg_U=Rk_Q_avg_U,
        om_bar_U=om_bar_U,
        Rk_Q_avg_C=Rk_Q_avg_C,
        om_bar_C=om_bar_C,
        uc_S=uc_S,
        Lam_1_nom_bank_C=Lam_1_nom_bank_C,
        z_om_bar_U=z_om_bar_U,
        d_om_d_B_U=d_om_d_B_U,
        d_om_d_X_U=d_om_d_X_U,
        z_om_bar_C=z_om_bar_C,
        d_om_d_B_C=d_om_d_B_C,
        d_om_d_X_C=d_om_d_X_C,
        Lam_1_S=Lam_1_S,
        Lam_0_S=Lam_0_S,
        F_viol_U=F_viol_U,
        f_viol_U=f_viol_U,
        F_viol_C=F_viol_C,
        f_viol_C=f_viol_C,
        Lam_1_nom_S=Lam_1_nom_S,
        xi_U=xi_U,
        d_xi_d_om_U=d_xi_d_om_U,
        xi_C=xi_C,
        d_xi_d_om_C=d_xi_d_om_C,
        avg_principal_flow_debt_U=avg_principal_flow_debt_U,
        Psi_flow_U=Psi_flow_U,
        avg_principal_flow_debt_C=avg_principal_flow_debt_C,
        Psi_flow_C=Psi_flow_C,
        marg_principal_flow_debt_U=marg_principal_flow_debt_U,
        principal_payments_U=principal_payments_U,
        marg_principal_flow_debt_C=marg_principal_flow_debt_C,
        principal_payments_C=principal_payments_C,
        net_outflow_debt_U=net_outflow_debt_U,
        net_outflow_debt_C=net_outflow_debt_C,
        net_outflow_financial_U=net_outflow_financial_U,
        net_outflow_financial_C=net_outflow_financial_C,
        D_U=D_U,
        D_C=D_C,
        uc_U=uc_U,
        uc_C=uc_C,
        D=D,
        Lam_1_U=Lam_1_U,
        Lam_0_U=Lam_0_U,
        Lam_1_C=Lam_1_C,
        Lam_0_C=Lam_0_C,
        y_check=y_check,
        Lam_1_nom_U=Lam_1_nom_U,
        Lam_1_nom_C=Lam_1_nom_C,
    )


def intermediate_variables_array(st):
    st = intermediate_variables(st)
    return np.array(
        (
            st.pi,
            st.Z,
            st.r_lag,
            st.marg_spread_flow_debt_S,
            st.bet_pi_S,
            st.r_new,
            st.r_bar,
            st.P_security_reg,
            st.share_C,
            st.K_U,
            st.after_tax_corp,
            st.r_cash_lag,
            st.psi_U,
            st.bet_U,
            st.lab_U,
            st.spread_at_issuance_U,
            st.liabilities_new_U,
            st.loans_new_U,
            st.loans_U,
            st.sig_om_U,
            st.kap_viol_U,
            st.tv_inv_efficiency,
            st.inv_cost_1_U,
            st.du_cash_U,
            st.psi_C,
            st.bet_C,
            st.lab_C,
            st.K_C,
            st.spread_at_issuance_C,
            st.liabilities_new_C,
            st.loans_new_C,
            st.loans_C,
            st.sig_om_C,
            st.kap_viol_C,
            st.inv_cost_1_C,
            st.du_cash_C,
            st.N_security_bank_C,
            st.share_afs_bank_C,
            st.deposits_new_bank_C,
            st.bet_pi_bank_C,
            st.spread_payments_received_C,
            st.net_deposit_flow_bank_C,
            st.marg_spread_flow_loan_bank_C,
            st.spread_L_implied_C,
            st.debt_issuance_U,
            st.dX_dX_U,
            st.debt_issuance_C,
            st.dX_dX_C,
            st.debt_base_payment,
            st.Om_denom_debt_S,
            st.nu_security,
            st.inv_U,
            st.marg_spread_flow_debt_U,
            st.marg_spread_flow_debt_C,
            st.net_outflow_cash_U,
            st.net_outflow_cash_C,
            st.bet_pi_U,
            st.y_U,
            st.mu_om_U,
            st.d_Phi_inv_U,
            st.inv_cost_0_U,
            st.bet_pi_C,
            st.lab,
            st.y_C,
            st.inv_C,
            st.K,
            st.required_capital_bank_C,
            st.loan_issuance_C,
            st.mu_om_C,
            st.d_Phi_inv_C,
            st.inv_cost_0_C,
            st.aoci_bank_C,
            st.V_afs_bank_C,
            st.mtm_share_bank_C,
            st.Om_denom_loan_bank_C,
            st.spread_payments_new_U,
            st.spread_payments_new_C,
            st.marg_principal_flow_debt_S,
            st.income_per_security,
            st.K_new_U,
            st.Om_denom_debt_U,
            st.fk_U,
            st.flab_U,
            st.Q_U,
            st.Phi_inv_U,
            st.Om_denom_debt_C,
            st.fk_C,
            st.flab_C,
            st.y,
            st.K_new_C,
            st.inv,
            st.K_share_U,
            st.net_loan_flow_bank_C,
            st.Q_C,
            st.Phi_inv_C,
            st.log_required_capital_no_aoci_bank_C,
            st.V_security_reg_bank_C,
            st.net_security_flow_bank_C,
            st.wage_bill_U,
            st.Q_bar_U,
            st.wage_bill_C,
            st.K_new,
            st.Q_bar_C,
            st.marg_loan_cost_bank_C,
            st.assets_bank_C,
            st.D_bank_C,
            st.ebitda_U,
            st.Rk_Q_marg_U,
            st.ebitda_C,
            st.wage_bill,
            st.Rk_Q_marg_C,
            st.marg_principal_flow_loan_bank_C,
            st.securities_to_assets_bank_C,
            st.capital_bank_C,
            st.uc_bank_C,
            st.x_U,
            st.ebitda_smooth_new_U,
            st.x_C,
            st.ebitda_smooth_new_C,
            st.c_S,
            st.capital_buffer_bank_C,
            st.Lam_1_bank_C,
            st.Lam_0_bank_C,
            st.Rk_Q_avg_U,
            st.om_bar_U,
            st.Rk_Q_avg_C,
            st.om_bar_C,
            st.uc_S,
            st.Lam_1_nom_bank_C,
            st.z_om_bar_U,
            st.d_om_d_B_U,
            st.d_om_d_X_U,
            st.z_om_bar_C,
            st.d_om_d_B_C,
            st.d_om_d_X_C,
            st.Lam_1_S,
            st.Lam_0_S,
            st.F_viol_U,
            st.f_viol_U,
            st.F_viol_C,
            st.f_viol_C,
            st.Lam_1_nom_S,
            st.xi_U,
            st.d_xi_d_om_U,
            st.xi_C,
            st.d_xi_d_om_C,
            st.avg_principal_flow_debt_U,
            st.Psi_flow_U,
            st.avg_principal_flow_debt_C,
            st.Psi_flow_C,
            st.marg_principal_flow_debt_U,
            st.principal_payments_U,
            st.marg_principal_flow_debt_C,
            st.principal_payments_C,
            st.net_outflow_debt_U,
            st.net_outflow_debt_C,
            st.net_outflow_financial_U,
            st.net_outflow_financial_C,
            st.D_U,
            st.D_C,
            st.uc_U,
            st.uc_C,
            st.D,
            st.Lam_1_U,
            st.Lam_0_U,
            st.Lam_1_C,
            st.Lam_0_C,
            st.y_check,
            st.Lam_1_nom_U,
            st.Lam_1_nom_C,
        )
    )


def read_expectations_variables(st):
    return _upd(
        st,
    )


def read_expectations_variables_array(st):
    st = read_expectations_variables(st)
    return np.array(())
