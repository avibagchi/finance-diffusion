# CSV Data Files

This directory contains CSV exports of the all_factors dataset.

## Files

### Main Data Files

1. **returns.csv** (11 × 500)
   - Daily returns for each asset
   - Index: date
   - Columns: asset names
   - Values: decimal returns (0.01 = 1%)

2. **prices.csv** (11 × 500)
   - Daily prices for each asset
   - Index: date
   - Columns: asset names
   - Values: price levels

3. **factors_long.csv** (5500 × 377)
   - All factors in long/stacked format
   - Columns: date, asset, me, market_equity, div12m_me, chcsho_12m, eqnpo_12m, ret_1_0, ret_3_1, ret_6_1, ret_9_1, ret_12_1, ret_12_7, ret_60_12, seas_1_1an, seas_1_1na, seas_2_5an, seas_2_5na, seas_6_10an, seas_6_10na, seas_11_15an, seas_11_15na, seas_16_20an, seas_16_20na, at_gr1, sale_gr1, capx_gr1, inv_gr1, debt_gr3, sale_gr3, capx_gr3, inv_gr1a, lti_gr1a, sti_gr1a, coa_gr1a, col_gr1a, cowc_gr1a, ncoa_gr1a, ncol_gr1a, nncoa_gr1a, fnl_gr1a, nfna_gr1a, tax_gr1a, be_gr1a, ebit_sale, gp_at, cop_at, ope_be, ni_be, ebit_bev, netis_at, eqnetis_at, dbnetis_at, oaccruals_at, oaccruals_ni, taccruals_at, taccruals_ni, noa_at, opex_at, at_turnover, sale_bev, rd_sale, cash_at, sale_emp_gr1, emp_gr1, ni_inc8q, noa_gr1a, ppeinv_gr1a, lnoa_gr1a, capx_gr2, saleq_gr1, niq_be, niq_at, niq_be_chg1, niq_at_chg1, rd5_at, dsale_dinv, dsale_drec, dgp_dsale, dsale_dsga, saleq_su, niq_su, capex_abn, op_atl1, gp_atl1, ope_bel1, cop_atl1, pi_nix, ocf_at, op_at, ocf_at_chg1, at_be, ocfq_saleq_std, tangibility, earnings_variability, aliq_at, f_score, o_score, z_score, kz_index, ni_ar1, ni_ivol, at_me, be_me, debt_me, netdebt_me, sale_me, ni_me, ocf_me, fcf_me, eqpo_me, eqnpo_me, rd_me, ival_me, bev_mev, ebitda_mev, aliq_mat, eq_dur, beta_60m, resff3_12_1, resff3_6_1, mispricing_mgmt, mispricing_perf, betabab_1260d, rmax5_rvol_21d, age, qmj, qmj_prof, qmj_growth, qmj_safety, enterprise_value, book_equity, assets, sales, net_income, div1m_me, div3m_me, div6m_me, divspc1m_me, divspc12m_me, chcsho_1m, chcsho_3m, chcsho_6m, eqnpo_1m, eqnpo_3m, eqnpo_6m, ret_2_0, ret_3_0, ret_6_0, ret_9_0, ret_12_0, ret_18_1, ret_24_1, ret_24_12, ret_36_1, ret_36_12, ret_48_12, ret_48_1, ret_60_1, ret_60_36, ca_gr1, nca_gr1, lt_gr1, cl_gr1, ncl_gr1, be_gr1, pstk_gr1, debt_gr1, cogs_gr1, sga_gr1, opex_gr1, at_gr3, ca_gr3, nca_gr3, lt_gr3, cl_gr3, ncl_gr3, be_gr3, pstk_gr3, cogs_gr3, sga_gr3, opex_gr3, cash_gr1a, rec_gr1a, ppeg_gr1a, intan_gr1a, debtst_gr1a, ap_gr1a, txp_gr1a, debtlt_gr1a, txditc_gr1a, oa_gr1a, ol_gr1a, fna_gr1a, gp_gr1a, ebitda_gr1a, ebit_gr1a, ope_gr1a, ni_gr1a, nix_gr1a, dp_gr1a, fincf_gr1a, ocf_gr1a, fcf_gr1a, nwc_gr1a, eqnetis_gr1a, dltnetis_gr1a, dstnetis_gr1a, dbnetis_gr1a, netis_gr1a, eqnpo_gr1a, eqbb_gr1a, eqis_gr1a, div_gr1a, eqpo_gr1a, capx_gr1a, cash_gr3a, inv_gr3a, rec_gr3a, ppeg_gr3a, lti_gr3a, intan_gr3a, debtst_gr3a, ap_gr3a, txp_gr3a, debtlt_gr3a, txditc_gr3a, coa_gr3a, col_gr3a, cowc_gr3a, ncoa_gr3a, ncol_gr3a, nncoa_gr3a, oa_gr3a, ol_gr3a, fna_gr3a, fnl_gr3a, nfna_gr3a, gp_gr3a, ebitda_gr3a, ebit_gr3a, ope_gr3a, ni_gr3a, nix_gr3a, dp_gr3a, fincf_gr3a, ocf_gr3a, fcf_gr3a, nwc_gr3a, eqnetis_gr3a, dltnetis_gr3a, dstnetis_gr3a, dbnetis_gr3a, netis_gr3a, eqnpo_gr3a, tax_gr3a, eqbb_gr3a, eqis_gr3a, div_gr3a, eqpo_gr3a, capx_gr3a, capx_at, rd_at, spi_at, xido_at, nri_at, gp_sale, ebitda_sale, pi_sale, ni_sale, nix_sale, ocf_sale, fcf_sale, ebitda_at, ebit_at, fi_at, ni_at, nix_be, ocf_be, fcf_be, gp_bev, ebitda_bev, fi_bev, cop_bev, gp_ppen, ebitda_ppen, fcf_ppen, fincf_at, eqis_at, dltnetis_at, dstnetis_at, eqnpo_at, eqbb_at, div_at, be_bev, debt_bev, cash_bev, pstk_bev, debtlt_bev, debtst_bev, int_debt, int_debtlt, ebitda_debt, profit_cl, ocf_cl, ocf_debt, cash_lt, inv_act, rec_act, debtst_debt, cl_lt, debtlt_debt, lt_ppen, debtlt_be, nwc_at, fcf_ocf, debt_at, debt_be, ebit_int, inv_days, rec_days, ap_days, cash_conversion, cash_cl, caliq_cl, ca_cl, inv_turnover, rec_turnover, ap_turnover, adv_sale, staff_sale, sale_be, div_ni, sale_nwc, tax_pi, ni_emp, sale_emp, niq_saleq_std, roeq_be_std, roe_be_std, intrinsic_value, gpoa_ch5, roe_ch5, roa_ch5, cfoa_ch5, gmar_ch5, cash_me, gp_me, ebitda_me, ebit_me, ope_me, nix_me, cop_me, div_me, eqbb_me, eqis_me, eqnetis_me, at_mev, ppen_mev, be_mev, cash_mev, sale_mev, gp_mev, ebit_mev, cop_mev, ocf_mev, fcf_mev, debt_mev, pstk_mev, debtlt_mev, debtst_mev, dltnetis_mev, dstnetis_mev, dbnetis_mev, netis_mev, fincf_mev, ivol_capm_60m
   - More compact than wide format

### Factor Files (Wide Format)

Individual CSV files for each factor (11 × 500):
- **factor_me.csv** - me values
- **factor_market_equity.csv** - market_equity values
- **factor_div12m_me.csv** - div12m_me values
- **factor_chcsho_12m.csv** - chcsho_12m values
- **factor_eqnpo_12m.csv** - eqnpo_12m values
- **factor_ret_1_0.csv** - ret_1_0 values
- **factor_ret_3_1.csv** - ret_3_1 values
- **factor_ret_6_1.csv** - ret_6_1 values
- **factor_ret_9_1.csv** - ret_9_1 values
- **factor_ret_12_1.csv** - ret_12_1 values
- **factor_ret_12_7.csv** - ret_12_7 values
- **factor_ret_60_12.csv** - ret_60_12 values
- **factor_seas_1_1an.csv** - seas_1_1an values
- **factor_seas_1_1na.csv** - seas_1_1na values
- **factor_seas_2_5an.csv** - seas_2_5an values
- **factor_seas_2_5na.csv** - seas_2_5na values
- **factor_seas_6_10an.csv** - seas_6_10an values
- **factor_seas_6_10na.csv** - seas_6_10na values
- **factor_seas_11_15an.csv** - seas_11_15an values
- **factor_seas_11_15na.csv** - seas_11_15na values
- **factor_seas_16_20an.csv** - seas_16_20an values
- **factor_seas_16_20na.csv** - seas_16_20na values
- **factor_at_gr1.csv** - at_gr1 values
- **factor_sale_gr1.csv** - sale_gr1 values
- **factor_capx_gr1.csv** - capx_gr1 values
- **factor_inv_gr1.csv** - inv_gr1 values
- **factor_debt_gr3.csv** - debt_gr3 values
- **factor_sale_gr3.csv** - sale_gr3 values
- **factor_capx_gr3.csv** - capx_gr3 values
- **factor_inv_gr1a.csv** - inv_gr1a values
- **factor_lti_gr1a.csv** - lti_gr1a values
- **factor_sti_gr1a.csv** - sti_gr1a values
- **factor_coa_gr1a.csv** - coa_gr1a values
- **factor_col_gr1a.csv** - col_gr1a values
- **factor_cowc_gr1a.csv** - cowc_gr1a values
- **factor_ncoa_gr1a.csv** - ncoa_gr1a values
- **factor_ncol_gr1a.csv** - ncol_gr1a values
- **factor_nncoa_gr1a.csv** - nncoa_gr1a values
- **factor_fnl_gr1a.csv** - fnl_gr1a values
- **factor_nfna_gr1a.csv** - nfna_gr1a values
- **factor_tax_gr1a.csv** - tax_gr1a values
- **factor_be_gr1a.csv** - be_gr1a values
- **factor_ebit_sale.csv** - ebit_sale values
- **factor_gp_at.csv** - gp_at values
- **factor_cop_at.csv** - cop_at values
- **factor_ope_be.csv** - ope_be values
- **factor_ni_be.csv** - ni_be values
- **factor_ebit_bev.csv** - ebit_bev values
- **factor_netis_at.csv** - netis_at values
- **factor_eqnetis_at.csv** - eqnetis_at values
- **factor_dbnetis_at.csv** - dbnetis_at values
- **factor_oaccruals_at.csv** - oaccruals_at values
- **factor_oaccruals_ni.csv** - oaccruals_ni values
- **factor_taccruals_at.csv** - taccruals_at values
- **factor_taccruals_ni.csv** - taccruals_ni values
- **factor_noa_at.csv** - noa_at values
- **factor_opex_at.csv** - opex_at values
- **factor_at_turnover.csv** - at_turnover values
- **factor_sale_bev.csv** - sale_bev values
- **factor_rd_sale.csv** - rd_sale values
- **factor_cash_at.csv** - cash_at values
- **factor_sale_emp_gr1.csv** - sale_emp_gr1 values
- **factor_emp_gr1.csv** - emp_gr1 values
- **factor_ni_inc8q.csv** - ni_inc8q values
- **factor_noa_gr1a.csv** - noa_gr1a values
- **factor_ppeinv_gr1a.csv** - ppeinv_gr1a values
- **factor_lnoa_gr1a.csv** - lnoa_gr1a values
- **factor_capx_gr2.csv** - capx_gr2 values
- **factor_saleq_gr1.csv** - saleq_gr1 values
- **factor_niq_be.csv** - niq_be values
- **factor_niq_at.csv** - niq_at values
- **factor_niq_be_chg1.csv** - niq_be_chg1 values
- **factor_niq_at_chg1.csv** - niq_at_chg1 values
- **factor_rd5_at.csv** - rd5_at values
- **factor_dsale_dinv.csv** - dsale_dinv values
- **factor_dsale_drec.csv** - dsale_drec values
- **factor_dgp_dsale.csv** - dgp_dsale values
- **factor_dsale_dsga.csv** - dsale_dsga values
- **factor_saleq_su.csv** - saleq_su values
- **factor_niq_su.csv** - niq_su values
- **factor_capex_abn.csv** - capex_abn values
- **factor_op_atl1.csv** - op_atl1 values
- **factor_gp_atl1.csv** - gp_atl1 values
- **factor_ope_bel1.csv** - ope_bel1 values
- **factor_cop_atl1.csv** - cop_atl1 values
- **factor_pi_nix.csv** - pi_nix values
- **factor_ocf_at.csv** - ocf_at values
- **factor_op_at.csv** - op_at values
- **factor_ocf_at_chg1.csv** - ocf_at_chg1 values
- **factor_at_be.csv** - at_be values
- **factor_ocfq_saleq_std.csv** - ocfq_saleq_std values
- **factor_tangibility.csv** - tangibility values
- **factor_earnings_variability.csv** - earnings_variability values
- **factor_aliq_at.csv** - aliq_at values
- **factor_f_score.csv** - f_score values
- **factor_o_score.csv** - o_score values
- **factor_z_score.csv** - z_score values
- **factor_kz_index.csv** - kz_index values
- **factor_ni_ar1.csv** - ni_ar1 values
- **factor_ni_ivol.csv** - ni_ivol values
- **factor_at_me.csv** - at_me values
- **factor_be_me.csv** - be_me values
- **factor_debt_me.csv** - debt_me values
- **factor_netdebt_me.csv** - netdebt_me values
- **factor_sale_me.csv** - sale_me values
- **factor_ni_me.csv** - ni_me values
- **factor_ocf_me.csv** - ocf_me values
- **factor_fcf_me.csv** - fcf_me values
- **factor_eqpo_me.csv** - eqpo_me values
- **factor_eqnpo_me.csv** - eqnpo_me values
- **factor_rd_me.csv** - rd_me values
- **factor_ival_me.csv** - ival_me values
- **factor_bev_mev.csv** - bev_mev values
- **factor_ebitda_mev.csv** - ebitda_mev values
- **factor_aliq_mat.csv** - aliq_mat values
- **factor_eq_dur.csv** - eq_dur values
- **factor_beta_60m.csv** - beta_60m values
- **factor_resff3_12_1.csv** - resff3_12_1 values
- **factor_resff3_6_1.csv** - resff3_6_1 values
- **factor_mispricing_mgmt.csv** - mispricing_mgmt values
- **factor_mispricing_perf.csv** - mispricing_perf values
- **factor_betabab_1260d.csv** - betabab_1260d values
- **factor_rmax5_rvol_21d.csv** - rmax5_rvol_21d values
- **factor_age.csv** - age values
- **factor_qmj.csv** - qmj values
- **factor_qmj_prof.csv** - qmj_prof values
- **factor_qmj_growth.csv** - qmj_growth values
- **factor_qmj_safety.csv** - qmj_safety values
- **factor_enterprise_value.csv** - enterprise_value values
- **factor_book_equity.csv** - book_equity values
- **factor_assets.csv** - assets values
- **factor_sales.csv** - sales values
- **factor_net_income.csv** - net_income values
- **factor_div1m_me.csv** - div1m_me values
- **factor_div3m_me.csv** - div3m_me values
- **factor_div6m_me.csv** - div6m_me values
- **factor_divspc1m_me.csv** - divspc1m_me values
- **factor_divspc12m_me.csv** - divspc12m_me values
- **factor_chcsho_1m.csv** - chcsho_1m values
- **factor_chcsho_3m.csv** - chcsho_3m values
- **factor_chcsho_6m.csv** - chcsho_6m values
- **factor_eqnpo_1m.csv** - eqnpo_1m values
- **factor_eqnpo_3m.csv** - eqnpo_3m values
- **factor_eqnpo_6m.csv** - eqnpo_6m values
- **factor_ret_2_0.csv** - ret_2_0 values
- **factor_ret_3_0.csv** - ret_3_0 values
- **factor_ret_6_0.csv** - ret_6_0 values
- **factor_ret_9_0.csv** - ret_9_0 values
- **factor_ret_12_0.csv** - ret_12_0 values
- **factor_ret_18_1.csv** - ret_18_1 values
- **factor_ret_24_1.csv** - ret_24_1 values
- **factor_ret_24_12.csv** - ret_24_12 values
- **factor_ret_36_1.csv** - ret_36_1 values
- **factor_ret_36_12.csv** - ret_36_12 values
- **factor_ret_48_12.csv** - ret_48_12 values
- **factor_ret_48_1.csv** - ret_48_1 values
- **factor_ret_60_1.csv** - ret_60_1 values
- **factor_ret_60_36.csv** - ret_60_36 values
- **factor_ca_gr1.csv** - ca_gr1 values
- **factor_nca_gr1.csv** - nca_gr1 values
- **factor_lt_gr1.csv** - lt_gr1 values
- **factor_cl_gr1.csv** - cl_gr1 values
- **factor_ncl_gr1.csv** - ncl_gr1 values
- **factor_be_gr1.csv** - be_gr1 values
- **factor_pstk_gr1.csv** - pstk_gr1 values
- **factor_debt_gr1.csv** - debt_gr1 values
- **factor_cogs_gr1.csv** - cogs_gr1 values
- **factor_sga_gr1.csv** - sga_gr1 values
- **factor_opex_gr1.csv** - opex_gr1 values
- **factor_at_gr3.csv** - at_gr3 values
- **factor_ca_gr3.csv** - ca_gr3 values
- **factor_nca_gr3.csv** - nca_gr3 values
- **factor_lt_gr3.csv** - lt_gr3 values
- **factor_cl_gr3.csv** - cl_gr3 values
- **factor_ncl_gr3.csv** - ncl_gr3 values
- **factor_be_gr3.csv** - be_gr3 values
- **factor_pstk_gr3.csv** - pstk_gr3 values
- **factor_cogs_gr3.csv** - cogs_gr3 values
- **factor_sga_gr3.csv** - sga_gr3 values
- **factor_opex_gr3.csv** - opex_gr3 values
- **factor_cash_gr1a.csv** - cash_gr1a values
- **factor_rec_gr1a.csv** - rec_gr1a values
- **factor_ppeg_gr1a.csv** - ppeg_gr1a values
- **factor_intan_gr1a.csv** - intan_gr1a values
- **factor_debtst_gr1a.csv** - debtst_gr1a values
- **factor_ap_gr1a.csv** - ap_gr1a values
- **factor_txp_gr1a.csv** - txp_gr1a values
- **factor_debtlt_gr1a.csv** - debtlt_gr1a values
- **factor_txditc_gr1a.csv** - txditc_gr1a values
- **factor_oa_gr1a.csv** - oa_gr1a values
- **factor_ol_gr1a.csv** - ol_gr1a values
- **factor_fna_gr1a.csv** - fna_gr1a values
- **factor_gp_gr1a.csv** - gp_gr1a values
- **factor_ebitda_gr1a.csv** - ebitda_gr1a values
- **factor_ebit_gr1a.csv** - ebit_gr1a values
- **factor_ope_gr1a.csv** - ope_gr1a values
- **factor_ni_gr1a.csv** - ni_gr1a values
- **factor_nix_gr1a.csv** - nix_gr1a values
- **factor_dp_gr1a.csv** - dp_gr1a values
- **factor_fincf_gr1a.csv** - fincf_gr1a values
- **factor_ocf_gr1a.csv** - ocf_gr1a values
- **factor_fcf_gr1a.csv** - fcf_gr1a values
- **factor_nwc_gr1a.csv** - nwc_gr1a values
- **factor_eqnetis_gr1a.csv** - eqnetis_gr1a values
- **factor_dltnetis_gr1a.csv** - dltnetis_gr1a values
- **factor_dstnetis_gr1a.csv** - dstnetis_gr1a values
- **factor_dbnetis_gr1a.csv** - dbnetis_gr1a values
- **factor_netis_gr1a.csv** - netis_gr1a values
- **factor_eqnpo_gr1a.csv** - eqnpo_gr1a values
- **factor_eqbb_gr1a.csv** - eqbb_gr1a values
- **factor_eqis_gr1a.csv** - eqis_gr1a values
- **factor_div_gr1a.csv** - div_gr1a values
- **factor_eqpo_gr1a.csv** - eqpo_gr1a values
- **factor_capx_gr1a.csv** - capx_gr1a values
- **factor_cash_gr3a.csv** - cash_gr3a values
- **factor_inv_gr3a.csv** - inv_gr3a values
- **factor_rec_gr3a.csv** - rec_gr3a values
- **factor_ppeg_gr3a.csv** - ppeg_gr3a values
- **factor_lti_gr3a.csv** - lti_gr3a values
- **factor_intan_gr3a.csv** - intan_gr3a values
- **factor_debtst_gr3a.csv** - debtst_gr3a values
- **factor_ap_gr3a.csv** - ap_gr3a values
- **factor_txp_gr3a.csv** - txp_gr3a values
- **factor_debtlt_gr3a.csv** - debtlt_gr3a values
- **factor_txditc_gr3a.csv** - txditc_gr3a values
- **factor_coa_gr3a.csv** - coa_gr3a values
- **factor_col_gr3a.csv** - col_gr3a values
- **factor_cowc_gr3a.csv** - cowc_gr3a values
- **factor_ncoa_gr3a.csv** - ncoa_gr3a values
- **factor_ncol_gr3a.csv** - ncol_gr3a values
- **factor_nncoa_gr3a.csv** - nncoa_gr3a values
- **factor_oa_gr3a.csv** - oa_gr3a values
- **factor_ol_gr3a.csv** - ol_gr3a values
- **factor_fna_gr3a.csv** - fna_gr3a values
- **factor_fnl_gr3a.csv** - fnl_gr3a values
- **factor_nfna_gr3a.csv** - nfna_gr3a values
- **factor_gp_gr3a.csv** - gp_gr3a values
- **factor_ebitda_gr3a.csv** - ebitda_gr3a values
- **factor_ebit_gr3a.csv** - ebit_gr3a values
- **factor_ope_gr3a.csv** - ope_gr3a values
- **factor_ni_gr3a.csv** - ni_gr3a values
- **factor_nix_gr3a.csv** - nix_gr3a values
- **factor_dp_gr3a.csv** - dp_gr3a values
- **factor_fincf_gr3a.csv** - fincf_gr3a values
- **factor_ocf_gr3a.csv** - ocf_gr3a values
- **factor_fcf_gr3a.csv** - fcf_gr3a values
- **factor_nwc_gr3a.csv** - nwc_gr3a values
- **factor_eqnetis_gr3a.csv** - eqnetis_gr3a values
- **factor_dltnetis_gr3a.csv** - dltnetis_gr3a values
- **factor_dstnetis_gr3a.csv** - dstnetis_gr3a values
- **factor_dbnetis_gr3a.csv** - dbnetis_gr3a values
- **factor_netis_gr3a.csv** - netis_gr3a values
- **factor_eqnpo_gr3a.csv** - eqnpo_gr3a values
- **factor_tax_gr3a.csv** - tax_gr3a values
- **factor_eqbb_gr3a.csv** - eqbb_gr3a values
- **factor_eqis_gr3a.csv** - eqis_gr3a values
- **factor_div_gr3a.csv** - div_gr3a values
- **factor_eqpo_gr3a.csv** - eqpo_gr3a values
- **factor_capx_gr3a.csv** - capx_gr3a values
- **factor_capx_at.csv** - capx_at values
- **factor_rd_at.csv** - rd_at values
- **factor_spi_at.csv** - spi_at values
- **factor_xido_at.csv** - xido_at values
- **factor_nri_at.csv** - nri_at values
- **factor_gp_sale.csv** - gp_sale values
- **factor_ebitda_sale.csv** - ebitda_sale values
- **factor_pi_sale.csv** - pi_sale values
- **factor_ni_sale.csv** - ni_sale values
- **factor_nix_sale.csv** - nix_sale values
- **factor_ocf_sale.csv** - ocf_sale values
- **factor_fcf_sale.csv** - fcf_sale values
- **factor_ebitda_at.csv** - ebitda_at values
- **factor_ebit_at.csv** - ebit_at values
- **factor_fi_at.csv** - fi_at values
- **factor_ni_at.csv** - ni_at values
- **factor_nix_be.csv** - nix_be values
- **factor_ocf_be.csv** - ocf_be values
- **factor_fcf_be.csv** - fcf_be values
- **factor_gp_bev.csv** - gp_bev values
- **factor_ebitda_bev.csv** - ebitda_bev values
- **factor_fi_bev.csv** - fi_bev values
- **factor_cop_bev.csv** - cop_bev values
- **factor_gp_ppen.csv** - gp_ppen values
- **factor_ebitda_ppen.csv** - ebitda_ppen values
- **factor_fcf_ppen.csv** - fcf_ppen values
- **factor_fincf_at.csv** - fincf_at values
- **factor_eqis_at.csv** - eqis_at values
- **factor_dltnetis_at.csv** - dltnetis_at values
- **factor_dstnetis_at.csv** - dstnetis_at values
- **factor_eqnpo_at.csv** - eqnpo_at values
- **factor_eqbb_at.csv** - eqbb_at values
- **factor_div_at.csv** - div_at values
- **factor_be_bev.csv** - be_bev values
- **factor_debt_bev.csv** - debt_bev values
- **factor_cash_bev.csv** - cash_bev values
- **factor_pstk_bev.csv** - pstk_bev values
- **factor_debtlt_bev.csv** - debtlt_bev values
- **factor_debtst_bev.csv** - debtst_bev values
- **factor_int_debt.csv** - int_debt values
- **factor_int_debtlt.csv** - int_debtlt values
- **factor_ebitda_debt.csv** - ebitda_debt values
- **factor_profit_cl.csv** - profit_cl values
- **factor_ocf_cl.csv** - ocf_cl values
- **factor_ocf_debt.csv** - ocf_debt values
- **factor_cash_lt.csv** - cash_lt values
- **factor_inv_act.csv** - inv_act values
- **factor_rec_act.csv** - rec_act values
- **factor_debtst_debt.csv** - debtst_debt values
- **factor_cl_lt.csv** - cl_lt values
- **factor_debtlt_debt.csv** - debtlt_debt values
- **factor_lt_ppen.csv** - lt_ppen values
- **factor_debtlt_be.csv** - debtlt_be values
- **factor_nwc_at.csv** - nwc_at values
- **factor_fcf_ocf.csv** - fcf_ocf values
- **factor_debt_at.csv** - debt_at values
- **factor_debt_be.csv** - debt_be values
- **factor_ebit_int.csv** - ebit_int values
- **factor_inv_days.csv** - inv_days values
- **factor_rec_days.csv** - rec_days values
- **factor_ap_days.csv** - ap_days values
- **factor_cash_conversion.csv** - cash_conversion values
- **factor_cash_cl.csv** - cash_cl values
- **factor_caliq_cl.csv** - caliq_cl values
- **factor_ca_cl.csv** - ca_cl values
- **factor_inv_turnover.csv** - inv_turnover values
- **factor_rec_turnover.csv** - rec_turnover values
- **factor_ap_turnover.csv** - ap_turnover values
- **factor_adv_sale.csv** - adv_sale values
- **factor_staff_sale.csv** - staff_sale values
- **factor_sale_be.csv** - sale_be values
- **factor_div_ni.csv** - div_ni values
- **factor_sale_nwc.csv** - sale_nwc values
- **factor_tax_pi.csv** - tax_pi values
- **factor_ni_emp.csv** - ni_emp values
- **factor_sale_emp.csv** - sale_emp values
- **factor_niq_saleq_std.csv** - niq_saleq_std values
- **factor_roeq_be_std.csv** - roeq_be_std values
- **factor_roe_be_std.csv** - roe_be_std values
- **factor_intrinsic_value.csv** - intrinsic_value values
- **factor_gpoa_ch5.csv** - gpoa_ch5 values
- **factor_roe_ch5.csv** - roe_ch5 values
- **factor_roa_ch5.csv** - roa_ch5 values
- **factor_cfoa_ch5.csv** - cfoa_ch5 values
- **factor_gmar_ch5.csv** - gmar_ch5 values
- **factor_cash_me.csv** - cash_me values
- **factor_gp_me.csv** - gp_me values
- **factor_ebitda_me.csv** - ebitda_me values
- **factor_ebit_me.csv** - ebit_me values
- **factor_ope_me.csv** - ope_me values
- **factor_nix_me.csv** - nix_me values
- **factor_cop_me.csv** - cop_me values
- **factor_div_me.csv** - div_me values
- **factor_eqbb_me.csv** - eqbb_me values
- **factor_eqis_me.csv** - eqis_me values
- **factor_eqnetis_me.csv** - eqnetis_me values
- **factor_at_mev.csv** - at_mev values
- **factor_ppen_mev.csv** - ppen_mev values
- **factor_be_mev.csv** - be_mev values
- **factor_cash_mev.csv** - cash_mev values
- **factor_sale_mev.csv** - sale_mev values
- **factor_gp_mev.csv** - gp_mev values
- **factor_ebit_mev.csv** - ebit_mev values
- **factor_cop_mev.csv** - cop_mev values
- **factor_ocf_mev.csv** - ocf_mev values
- **factor_fcf_mev.csv** - fcf_mev values
- **factor_debt_mev.csv** - debt_mev values
- **factor_pstk_mev.csv** - pstk_mev values
- **factor_debtlt_mev.csv** - debtlt_mev values
- **factor_debtst_mev.csv** - debtst_mev values
- **factor_dltnetis_mev.csv** - dltnetis_mev values
- **factor_dstnetis_mev.csv** - dstnetis_mev values
- **factor_dbnetis_mev.csv** - dbnetis_mev values
- **factor_netis_mev.csv** - netis_mev values
- **factor_fincf_mev.csv** - fincf_mev values
- **factor_ivol_capm_60m.csv** - ivol_capm_60m values

### Summary Statistics

4. **summary_stats.csv**
   - Mean, std, min, max, median for all variables
   - Useful for quick data inspection

## Data Period

- Start: 2010-01-04
- End: 2010-01-15
- Days: 11
- Assets: 500
- Factors: 375

## Loading CSVs

### Python (pandas)

```python
import pandas as pd

# Load returns
returns = pd.read_csv('returns.csv', index_col='date', parse_dates=True)

# Load prices
prices = pd.read_csv('prices.csv', index_col='date', parse_dates=True)

# Load factors (long format)
factors = pd.read_csv('factors_long.csv', parse_dates=['date'])

# Load a specific factor (wide format)
factor_ret_1_0 = pd.read_csv('factor_ret_1_0.csv', index_col='date', parse_dates=True)

# Load summary stats
stats = pd.read_csv('summary_stats.csv')
```

### R

```r
# Load returns
returns <- read.csv('returns.csv', row.names=1)

# Load prices
prices <- read.csv('prices.csv', row.names=1)

# Load factors (long format)
factors <- read.csv('factors_long.csv')

# Load summary stats
stats <- read.csv('summary_stats.csv')
```

### Excel / Google Sheets

All CSV files can be directly opened in Excel or Google Sheets for viewing and analysis.

## Notes

- All CSV files use comma (`,`) as delimiter
- Dates are in YYYY-MM-DD format
- Missing values (if any) are represented as empty cells
- Files are UTF-8 encoded
