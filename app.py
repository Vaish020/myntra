"""
app.py — AURA India Analytics Dashboard v2
===========================================
Creative, unique, premium dark dashboard.
4 analysis layers: Descriptive | Diagnostic | Predictive | Prescriptive
"""

import streamlit as st

st.set_page_config(
    page_title="AURA India · Analytics",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AURA India — Art Experience Pod Analytics"}
)

import pandas as pd
import numpy as np

from aura_theme import (GLOBAL_CSS, GOLD, TEAL, ORANGE, ROSE, MUTED,
                         SURFACE, SURFACE2, SURFACE3, INK, BG, INDIGO,
                         BORDER, _hex_to_rgb)
from aura_data import (load_data, train_classification_models,
                        train_regression_models, train_clustering)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:

    # Logo
    st.markdown(f"""
    <div style="padding:28px 20px 20px;border-bottom:1px solid rgba(232,197,71,0.08);
                margin-bottom:20px;position:relative;">
        <div style="position:absolute;top:20px;right:16px;font-size:42px;font-weight:900;
                    color:{GOLD};opacity:0.04;line-height:1;font-family:'Outfit',sans-serif;
                    letter-spacing:-0.04em;">AURA</div>
        <div style="font-size:30px;font-weight:900;color:{GOLD};letter-spacing:0.05em;
                    line-height:1;font-family:'Outfit',sans-serif;">AURA</div>
        <div style="font-size:9px;letter-spacing:0.25em;text-transform:uppercase;
                    color:{MUTED};margin-top:4px;font-family:'JetBrains Mono',monospace;">
            India · Analytics</div>
        <div style="margin-top:10px;display:flex;gap:4px;">
            <div style="width:20px;height:2px;background:{GOLD};border-radius:1px;"></div>
            <div style="width:10px;height:2px;background:{TEAL};border-radius:1px;"></div>
            <div style="width:6px;height:2px;background:{ORANGE};border-radius:1px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Home
    if st.button("🏠  Dashboard Home", key="nav_home", use_container_width=True):
        st.session_state["selected_tab"] = "Home"
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Navigation sections
    nav_sections = [
        ("📊 Descriptive", TEAL,   "What does the data say?", [
            ("📊 Overview — Market Summary",           "Overview"),
        ]),
        ("🔍 Diagnostic",  ORANGE, "Why is it happening?",    [
            ("🔍 Diagnostic — Why Customers Convert",  "Diagnostic"),
        ]),
        ("🤖 Predictive",  INDIGO, "What will happen?",       [
            ("🧩 Clustering — Persona Segments",       "Clustering"),
            ("🎯 Classification — Interest Prediction","Classification"),
            ("🔗 Association Rules — Bundle Mining",   "ARM"),
            ("📈 Regression — WTP Prediction",         "Regression"),
        ]),
        ("🚀 Prescriptive",ROSE,   "What should we do?",      [
            ("🚀 Predict New Customers",               "Predict"),
        ]),
    ]

    for sec_label, color, sec_desc, tabs in nav_sections:
        st.markdown(f"""
        <div style="margin-top:14px;margin-bottom:4px;padding-left:4px;">
            <div style="display:flex;align-items:center;gap:6px;">
                <div style="width:2px;height:12px;background:{color};border-radius:1px;"></div>
                <span style="font-size:9px;font-weight:700;color:{color};letter-spacing:0.18em;
                             text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
                    {sec_label}</span>
            </div>
            <div style="font-size:9px;color:{MUTED};margin-left:8px;margin-top:1px;
                        font-style:italic;">{sec_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        for tab_label, tab_key in tabs:
            short = tab_label.split("—")[0].strip() if "—" in tab_label else tab_label
            active = st.session_state.get("selected_tab","Home") == tab_key
            btn_style = f"background:rgba({_hex_to_rgb(color)},0.1) !important;color:{color} !important;border-color:rgba({_hex_to_rgb(color)},0.2) !important;" if active else ""
            if active:
                st.markdown(f"""
                <div style="background:rgba({_hex_to_rgb(color)},0.09);
                            border:1px solid rgba({_hex_to_rgb(color)},0.2);
                            border-radius:3px;padding:7px 12px;margin-bottom:2px;
                            font-size:11px;font-weight:600;color:{color};cursor:default;">
                    {short}</div>
                """, unsafe_allow_html=True)
            else:
                if st.button(short, key=f"nav_{tab_key}", use_container_width=True):
                    st.session_state["selected_tab"] = tab_key
                    st.rerun()

    # Footer stats
    st.markdown(f"""
    <div style="margin-top:24px;padding-top:16px;
                border-top:1px solid rgba(255,255,255,0.05);">
        <div style="font-size:9px;letter-spacing:0.15em;text-transform:uppercase;
                    color:{MUTED};margin-bottom:10px;
                    font-family:'JetBrains Mono',monospace;">Live Dataset</div>
        <div style="font-size:10px;color:{TEAL};margin-bottom:3px;">
            ● S1 · 2,000 respondents · 81 cols</div>
        <div style="font-size:10px;color:{TEAL};margin-bottom:3px;">
            ● S2 · 1,314 deep profiles · 18 cols</div>
        <div style="font-size:10px;color:{GOLD};margin-bottom:3px;">
            ● ARM · 2,000 transactions · 42 items</div>
        <div style="font-size:10px;color:{ORANGE};">
            ● Combined · 1,314 rows · 98 cols</div>
    </div>
    <div style="margin-top:20px;font-size:9px;color:rgba(122,114,104,0.4);
                font-family:'JetBrains Mono',monospace;">AURA India · 2026 · v2.0</div>
    """, unsafe_allow_html=True)

# ── LOAD DATA & MODELS ─────────────────────────────────────────
if "selected_tab" not in st.session_state:
    st.session_state["selected_tab"] = "Home"

with st.spinner("Initialising AURA analytics engine..."):
    df1, df2, arm, wide = load_data()

with st.spinner("Training models (cached after first run)..."):
    clf_models, clf_results, clf_feat_imp, X_test_clf, y_test_clf, X_train_clf, y_train_clf = \
        train_classification_models(df1)
    reg_models, reg_results, reg_feat_imp, X_test_reg, y_test_reg, reg_scaler = \
        train_regression_models(df1)
    km_model, df_clustered, km_scaler, best_k, k_range, inertias, silhouettes, pca = \
        train_clustering(df1)

selected = st.session_state.get("selected_tab", "Home")

# ═══════════════════════════════════════════════════════════════
# HOME PAGE — creative landing
# ═══════════════════════════════════════════════════════════════
if selected == "Home":

    # Hero
    interested_n = int((df1["aura_interest_label"] == "Interested").sum())
    median_wtp   = int(df1["session_wtp_numeric"].median())

    st.markdown(f"""
    <div style="position:relative;padding:56px 0 40px;overflow:hidden;
                border-bottom:1px solid rgba(232,197,71,0.07);margin-bottom:40px;">

        <!-- decorative grid -->
        <div style="position:absolute;inset:0;
            background-image:linear-gradient(rgba(232,197,71,0.03) 1px,transparent 1px),
                             linear-gradient(90deg,rgba(232,197,71,0.03) 1px,transparent 1px);
            background-size:48px 48px;pointer-events:none;"></div>

        <!-- decorative orbs -->
        <div style="position:absolute;top:-60px;right:-40px;width:320px;height:320px;
            border-radius:50%;
            background:radial-gradient(circle,rgba(94,196,161,0.06) 0%,transparent 70%);
            pointer-events:none;"></div>
        <div style="position:absolute;bottom:-40px;left:10%;width:240px;height:240px;
            border-radius:50%;
            background:radial-gradient(circle,rgba(232,197,71,0.05) 0%,transparent 70%);
            pointer-events:none;"></div>

        <div style="position:relative;z-index:2;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                        letter-spacing:0.25em;text-transform:uppercase;color:{TEAL};
                        margin-bottom:16px;display:flex;align-items:center;gap:10px;">
                <div style="width:28px;height:1px;background:{TEAL};"></div>
                Data Analytics · MGB · India Market Intelligence
            </div>

            <div style="font-family:'Outfit',sans-serif;font-size:clamp(40px,7vw,72px);
                        font-weight:900;line-height:0.92;letter-spacing:-0.03em;
                        margin-bottom:20px;">
                <span style="color:{INK};">ART PODS.</span><br>
                <span style="color:{GOLD};">DATA.</span>
                <span style="color:rgba(242,237,230,0.2);">DECISIONS.</span>
            </div>

            <p style="font-size:16px;color:{MUTED};max-width:560px;line-height:1.75;
                      margin-bottom:32px;font-weight:300;">
                A complete 4-layer analytics pipeline validating the
                <strong style="color:{INK};">AURA Art Experience Pod</strong> business
                concept for India — from raw survey data to prescriptive marketing actions.
            </p>

            <!-- Live stats strip -->
            <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:center;
                        padding:16px 24px;background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.05);border-radius:6px;
                        width:fit-content;">
                <div>
                    <div style="font-size:28px;font-weight:800;color:{GOLD};
                                font-family:'Outfit',sans-serif;line-height:1;">2,000</div>
                    <div style="font-size:9px;color:{MUTED};letter-spacing:0.15em;
                                text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
                        Respondents</div>
                </div>
                <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
                <div>
                    <div style="font-size:28px;font-weight:800;color:{TEAL};
                                font-family:'Outfit',sans-serif;line-height:1;">
                        {interested_n:,}</div>
                    <div style="font-size:9px;color:{MUTED};letter-spacing:0.15em;
                                text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
                        Interested (71.8%)</div>
                </div>
                <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
                <div>
                    <div style="font-size:28px;font-weight:800;color:{ORANGE};
                                font-family:'Outfit',sans-serif;line-height:1;">
                        ₹{median_wtp:,}</div>
                    <div style="font-size:9px;color:{MUTED};letter-spacing:0.15em;
                                text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
                        Median WTP</div>
                </div>
                <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
                <div>
                    <div style="font-size:28px;font-weight:800;color:{INDIGO};
                                font-family:'Outfit',sans-serif;line-height:1;">7</div>
                    <div style="font-size:9px;color:{MUTED};letter-spacing:0.15em;
                                text-transform:uppercase;font-family:'JetBrains Mono',monospace;">
                        ML Models</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 ANALYSIS CARDS ──────────────────────────────────────
    cols = st.columns(4, gap="small")

    card_data = [
        ("Overview",        "Overview",        "📊", TEAL,   "Descriptive",
         "What does the data say?",
         "10 interactive charts — demographics, demand signals, WTP distribution, art form rankings, spending behaviour patterns.",
         ["2,000 respondents", "10 chart types", "Demand signals"]),
        ("Diagnostic",      "Diagnostic",      "🔍", ORANGE, "Diagnostic",
         "Why are customers interested or not?",
         "Barrier analysis, conversion funnel, correlation heatmap, competitive substitutes, discount preference intelligence.",
         ["Barrier analysis", "Conversion funnel", "Correlation drivers"]),
        ("Clustering",      "Predictive",      "🤖", INDIGO, "Predictive",
         "Who are our customers & what will they do?",
         "K-Means personas · Random Forest + XGBoost classification · Apriori ARM bundle mining · WTP regression with live predictor.",
         ["K-Means + PCA", "3 classifiers + ROC", "ARM + Regression"]),
        ("Predict",         "Prescriptive",    "🚀", ROSE,   "Prescriptive",
         "Upload new data → instant scoring.",
         "Upload a CSV of new survey respondents. Get predicted interest, WTP, persona, and personalised marketing action for each.",
         ["Upload CSV", "Predict + score", "Download results"]),
    ]

    for col, (tab_key, type_label, icon, color, badge, question, desc, bullets) in zip(cols, card_data):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,{SURFACE2} 0%,{SURFACE3} 100%);
                        border:1px solid rgba(255,255,255,0.05);
                        border-top:2px solid {color};
                        border-radius:8px;padding:28px 22px;height:100%;
                        position:relative;overflow:hidden;
                        transition:transform 0.2s,border-color 0.2s;">

                <!-- bg number watermark -->
                <div style="position:absolute;bottom:-10px;right:12px;
                            font-size:80px;font-weight:900;color:{color};opacity:0.04;
                            line-height:1;font-family:'Outfit',sans-serif;">{icon}</div>

                <!-- badge -->
                <div style="display:inline-block;background:rgba({_hex_to_rgb(color)},0.1);
                            border:1px solid rgba({_hex_to_rgb(color)},0.2);
                            border-radius:2px;padding:2px 8px;margin-bottom:14px;">
                    <span style="font-size:8px;font-weight:700;color:{color};
                                 letter-spacing:0.18em;text-transform:uppercase;
                                 font-family:'JetBrains Mono',monospace;">{badge}</span>
                </div>

                <div style="font-size:32px;margin-bottom:8px;">{icon}</div>

                <div style="font-size:16px;font-weight:700;color:{INK};
                            margin-bottom:6px;line-height:1.2;">{question}</div>

                <div style="font-size:12px;color:{MUTED};line-height:1.65;
                            margin-bottom:16px;">{desc}</div>

                <div style="border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">
                    {''.join([
                        f'<div style="font-size:10px;color:{color};margin-bottom:4px;'
                        f'display:flex;align-items:center;gap:6px;">'
                        f'<span style="opacity:0.6;">▸</span>{b}</div>'
                        for b in bullets
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Open {badge} →", key=f"card_{tab_key}", use_container_width=True):
                st.session_state["selected_tab"] = tab_key
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PIPELINE STRIP ────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.05);
                border-radius:8px;padding:28px 32px;margin-bottom:24px;">

        <div style="font-size:9px;font-weight:700;color:{GOLD};letter-spacing:0.2em;
                    text-transform:uppercase;margin-bottom:20px;
                    font-family:'JetBrains Mono',monospace;">
            ◆ Analytics Pipeline Flow</div>

        <div style="display:flex;align-items:center;gap:0;flex-wrap:wrap;gap:0;">
            <div style="background:{TEAL};color:#0a0805;padding:11px 20px;
                        border-radius:4px 0 0 4px;font-size:11px;font-weight:700;
                        letter-spacing:0.08em;white-space:nowrap;">📊 DESCRIPTIVE</div>
            <div style="background:rgba(94,196,161,0.15);color:{TEAL};
                        padding:11px 14px;font-size:16px;">→</div>
            <div style="background:{ORANGE};color:#0a0805;padding:11px 20px;
                        font-size:11px;font-weight:700;letter-spacing:0.08em;
                        white-space:nowrap;">🔍 DIAGNOSTIC</div>
            <div style="background:rgba(224,124,58,0.15);color:{ORANGE};
                        padding:11px 14px;font-size:16px;">→</div>
            <div style="background:{INDIGO};color:#0a0805;padding:11px 20px;
                        font-size:11px;font-weight:700;letter-spacing:0.08em;
                        white-space:nowrap;">🤖 PREDICTIVE</div>
            <div style="background:rgba(123,159,232,0.15);color:{INDIGO};
                        padding:11px 14px;font-size:16px;">→</div>
            <div style="background:{ROSE};color:#0a0805;padding:11px 20px;
                        border-radius:0 4px 4px 0;font-size:11px;font-weight:700;
                        letter-spacing:0.08em;white-space:nowrap;">🚀 PRESCRIPTIVE</div>
        </div>

        <div style="display:flex;gap:0;margin-top:10px;flex-wrap:wrap;">
            <div style="padding:4px 20px;font-size:10px;color:{TEAL};min-width:140px;">
                Overview · Demand · WTP</div>
            <div style="padding:4px 14px;color:transparent;min-width:48px;">→</div>
            <div style="padding:4px 20px;font-size:10px;color:{ORANGE};min-width:140px;">
                Barriers · Funnel · Why</div>
            <div style="padding:4px 14px;color:transparent;min-width:48px;">→</div>
            <div style="padding:4px 20px;font-size:10px;color:{INDIGO};min-width:200px;">
                Cluster · Classify · ARM · Regress</div>
            <div style="padding:4px 14px;color:transparent;min-width:48px;">→</div>
            <div style="padding:4px 20px;font-size:10px;color:{ROSE};">
                Score · Recommend · Act</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── BOTTOM KPI ROW ────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Survey 1", "2,000 rows")
    k2.metric("Survey 2", "1,314 rows")
    k3.metric("Features", "81 columns")
    k4.metric("Interested", f"{interested_n:,} (71.8%)")
    k5.metric("Median WTP", f"₹{median_wtp:,}")
    k6.metric("ML Models", "7 trained")


# ═══════════════════════════════════════════════════════════════
# DESCRIPTIVE
# ═══════════════════════════════════════════════════════════════
elif selected == "Overview":
    from aura_theme import analysis_badge
    analysis_badge("descriptive")
    import tab_overview
    tab_overview.render(df1, df2, arm, wide)

# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════
elif selected == "Diagnostic":
    from aura_theme import analysis_badge
    analysis_badge("diagnostic")
    import tab_diagnostic
    tab_diagnostic.render(df1, df2, arm, wide)

# ═══════════════════════════════════════════════════════════════
# PREDICTIVE — CLUSTERING
# ═══════════════════════════════════════════════════════════════
elif selected == "Clustering":
    from aura_theme import analysis_badge
    analysis_badge("predictive")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.18em;
                text-transform:uppercase;color:{INDIGO};margin-bottom:20px;
                display:flex;align-items:center;gap:8px;">
        <div style="width:16px;height:1px;background:{INDIGO};"></div>
        Sub-layer 1 of 4 · K-Means Clustering</div>
    """, unsafe_allow_html=True)
    import tab_clustering
    tab_clustering.render(
        df1, df2, arm, wide,
        km_model, df_clustered, km_scaler,
        best_k, k_range, inertias, silhouettes, pca
    )

# ═══════════════════════════════════════════════════════════════
# PREDICTIVE — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════
elif selected == "Classification":
    from aura_theme import analysis_badge
    analysis_badge("predictive")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.18em;
                text-transform:uppercase;color:{INDIGO};margin-bottom:20px;
                display:flex;align-items:center;gap:8px;">
        <div style="width:16px;height:1px;background:{INDIGO};"></div>
        Sub-layer 2 of 4 · Classification Models</div>
    """, unsafe_allow_html=True)
    import tab_classification
    tab_classification.render(
        df1, df2, arm, wide,
        clf_models, clf_results, clf_feat_imp,
        X_test_clf, y_test_clf, X_train_clf, y_train_clf
    )

# ═══════════════════════════════════════════════════════════════
# PREDICTIVE — ARM
# ═══════════════════════════════════════════════════════════════
elif selected == "ARM":
    from aura_theme import analysis_badge
    analysis_badge("predictive")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.18em;
                text-transform:uppercase;color:{INDIGO};margin-bottom:20px;
                display:flex;align-items:center;gap:8px;">
        <div style="width:16px;height:1px;background:{INDIGO};"></div>
        Sub-layer 3 of 4 · Association Rule Mining</div>
    """, unsafe_allow_html=True)
    import tab_arm
    tab_arm.render(df1, df2, arm, wide)

# ═══════════════════════════════════════════════════════════════
# PREDICTIVE — REGRESSION
# ═══════════════════════════════════════════════════════════════
elif selected == "Regression":
    from aura_theme import analysis_badge
    analysis_badge("predictive")
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.18em;
                text-transform:uppercase;color:{INDIGO};margin-bottom:20px;
                display:flex;align-items:center;gap:8px;">
        <div style="width:16px;height:1px;background:{INDIGO};"></div>
        Sub-layer 4 of 4 · Regression Models</div>
    """, unsafe_allow_html=True)
    import tab_regression
    tab_regression.render(
        df1, df2, arm, wide,
        reg_models, reg_results, reg_feat_imp,
        X_test_reg, y_test_reg, reg_scaler
    )

# ═══════════════════════════════════════════════════════════════
# PRESCRIPTIVE
# ═══════════════════════════════════════════════════════════════
elif selected == "Predict":
    from aura_theme import analysis_badge
    analysis_badge("prescriptive")
    import tab_predict
    tab_predict.render(
        df1, df2, arm, wide,
        clf_models, reg_models, km_model, km_scaler
    )
