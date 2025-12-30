import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from indexer_client import DEFAULT_INDEX_PATTERN, WazuhIndexerClient
from ml_training import TrainingArtifacts, train_severity_models
from wazuh_client import WazuhManagerClient

st.set_page_config(page_title="Wazuh → Dataset ML", layout="wide")

PRIMARY_COLOR = "#076d27"

st.markdown(
    f"""
    <style>
    .main {{ background-color: #ffffff; }}
    .css-1d391kg {{ padding-top: 2rem; }}
    .kpi-card {{
        background: #ffffff;
        border: 1px solid rgba(7, 109, 39, 0.25);
        border-left: 6px solid {PRIMARY_COLOR};
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 12px;
        color: #1c1c1c;
    }}
    .kpi-label {{ font-size: 0.9rem; color: #4d4d4d; margin-bottom: 4px; }}
    .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {PRIMARY_COLOR}; }}
    .section-title {{ font-size: 1.1rem; font-weight: 700; color: {PRIMARY_COLOR}; margin-bottom: 0.4rem; }}
    .dataframe th {{ background: rgba(7, 109, 39, 0.1); }}
    </style>
    """,
    unsafe_allow_html=True,
)


def build_datetime_range(selected: List[date]) -> Optional[tuple[datetime, datetime]]:
    if not selected:
        return None
    if isinstance(selected, list) and len(selected) == 2:
        start_date, end_date = selected
    else:
        return None
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)
    return start_dt, end_dt


def normalize_filters_from_df(df: pd.DataFrame) -> None:
    st.session_state["available_agents"] = sorted(df["agent.name"].dropna().unique().tolist())
    st.session_state["available_mitre"] = sorted(
        {
            mitre
            for ids in df.get("rule.mitre.id", [])
            if isinstance(ids, list)
            for mitre in ids
        }
    )
    st.session_state["available_rules"] = sorted(df["rule.id"].dropna().astype(str).unique().tolist())
    st.session_state["available_levels"] = sorted(
        df["rule.level"].dropna().astype(int, errors="ignore").unique().tolist()
        if "rule.level" in df
        else []
    )


def prepare_dataframe(hits: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [WazuhIndexerClient.flatten_alert(hit) for hit in hits]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rule.mitre.id"] = df["rule.mitre.id"].apply(lambda vals: vals if isinstance(vals, list) else [])
    df["rule.mitre.id_str"] = df["rule.mitre.id"].apply(lambda vals: ";".join(vals))
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce")

    # Auto-étiquetage de la sévérité si rule.level est présent
    if "rule.level" in df.columns and "severity" not in df.columns:
        level_numeric = pd.to_numeric(df["rule.level"], errors="coerce")
        severity = pd.Series(["low"] * len(df), index=df.index)
        severity[level_numeric.between(0, 3, inclusive="both")] = "low"
        severity[level_numeric.between(4, 6, inclusive="both")] = "medium"
        severity[level_numeric.between(7, 10, inclusive="both")] = "high"
        severity[level_numeric.between(11, 15, inclusive="both")] = "critical"
        df["severity"] = severity
    return df


@st.cache_data(show_spinner=False)
def load_alerts(
    base_url: str,
    username: str,
    password: str,
    verify_tls: bool,
    start: datetime,
    end: datetime,
    agents: Optional[List[str]],
    mitre_ids: Optional[List[str]],
    rule_ids: Optional[List[str]],
    levels: Optional[List[int]],
    atomic_only: bool,
    limit: int,
) -> pd.DataFrame:
    client = WazuhIndexerClient(
        base_url=base_url,
        username=username,
        password=password,
        verify_tls=verify_tls,
    )
    hits = client.fetch_alerts(
        start=start,
        end=end,
        agents=agents or None,
        mitre_ids=mitre_ids or None,
        rule_ids=rule_ids or None,
        levels=levels or None,
        atomic_only=atomic_only,
        limit=limit,
    )
    return prepare_dataframe(hits)


def render_kpis(df: pd.DataFrame) -> None:
    alerts_count = len(df)
    agents_count = df["agent.name"].nunique()
    top_mitre = df["rule.mitre.id"].explode().value_counts().head(1)
    top_rule = df["rule.id"].value_counts().head(1)

    cols = st.columns(4)
    metrics = [
        ("Alertes", alerts_count),
        ("Agents", agents_count),
        ("Top MITRE", top_mitre.index[0] if not top_mitre.empty else "N/A"),
        ("Top Rule", top_rule.index[0] if not top_rule.empty else "N/A"),
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Aucune alerte trouvée pour ces filtres.")
        return

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("<div class='section-title'>Top techniques MITRE</div>", unsafe_allow_html=True)
        mitre_counts = df["rule.mitre.id"].explode().value_counts().reset_index()
        mitre_counts.columns = ["MITRE ID", "Total"]
        if not mitre_counts.empty:
            fig_mitre = px.bar(mitre_counts, x="MITRE ID", y="Total", color="Total", color_continuous_scale=[PRIMARY_COLOR, PRIMARY_COLOR])
            fig_mitre.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_mitre, use_container_width=True)
        else:
            st.caption("Aucune technique MITRE disponible.")

        st.markdown("<div class='section-title'>Volume d'alertes dans le temps</div>", unsafe_allow_html=True)
        time_series = df.set_index("@timestamp").resample("1H").size().reset_index(name="count")
        if not time_series.empty:
            fig_time = px.line(time_series, x="@timestamp", y="count", markers=True, color_discrete_sequence=[PRIMARY_COLOR])
            fig_time.update_layout(template="plotly_white")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.caption("Pas de données temporelles à afficher.")

    with chart_col2:
        st.markdown("<div class='section-title'>Répartition par agent</div>", unsafe_allow_html=True)
        agent_counts = df["agent.name"].value_counts().reset_index()
        agent_counts.columns = ["Agent", "Total"]
        if not agent_counts.empty:
            fig_agents = px.pie(agent_counts, names="Agent", values="Total", color_discrete_sequence=[PRIMARY_COLOR])
            fig_agents.update_layout(template="plotly_white")
            st.plotly_chart(fig_agents, use_container_width=True)
        else:
            st.caption("Pas d'agents à afficher.")

        st.markdown("<div class='section-title'>Top descriptions de règle</div>", unsafe_allow_html=True)
        rule_desc = df[["rule.description"]].fillna("Inconnue")
        rule_desc_counts = rule_desc.value_counts().reset_index(name="Total")
        if not rule_desc_counts.empty:
            fig_rules = px.bar(
                rule_desc_counts.head(10),
                x="rule.description",
                y="Total",
                color="Total",
                color_continuous_scale=[PRIMARY_COLOR, PRIMARY_COLOR],
            )
            fig_rules.update_layout(template="plotly_white")
            fig_rules.update_xaxes(tickangle=40)
            st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.caption("Aucune description de règle disponible.")


def run_connection_tests(manager_cfg: Dict[str, Any], indexer_cfg: Dict[str, Any]) -> None:
    st.subheader("Tests de connexion")
    try:
        manager = WazuhManagerClient(**manager_cfg)
        manager.authenticate()
        manager_status = manager.test_health()
        st.success("Wazuh Manager OK")
        st.json(manager_status)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Wazuh Manager KO : {exc}")

    try:
        indexer = WazuhIndexerClient(**indexer_cfg)
        status = indexer.test_connection()
        st.success("Wazuh Indexer OK")
        st.json(status)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Wazuh Indexer KO : {exc}")


def export_csv(
    indexer_cfg: Dict[str, Any],
    start: datetime,
    end: datetime,
    agents: Optional[List[str]],
    mitre_ids: Optional[List[str]],
    rule_ids: Optional[List[str]],
    levels: Optional[List[int]],
    atomic_only: bool,
) -> None:
    client = WazuhIndexerClient(**indexer_cfg)
    all_rows: List[Dict[str, Any]] = []
    progress = st.progress(0)
    batch_index = 0

    with st.spinner("Export en cours..."):
        for hits, _ in client.fetch_all_paginated(
            start=start,
            end=end,
            agents=agents,
            mitre_ids=mitre_ids,
            rule_ids=rule_ids,
            levels=levels,
            atomic_only=atomic_only,
        ):
            all_rows.extend([WazuhIndexerClient.flatten_alert(hit) for hit in hits])
            batch_index += 1
            progress.progress(min(1.0, 0.05 + batch_index * 0.05))

    if not all_rows:
        st.warning("Aucune donnée à exporter pour ces filtres.")
        return

    df_export = pd.DataFrame(all_rows)
    df_export["rule.mitre.id"] = df_export["rule.mitre.id"].apply(lambda vals: vals if isinstance(vals, list) else [])
    df_export["rule.mitre.id"] = df_export["rule.mitre.id"].apply(lambda vals: ";".join(vals))

    os.makedirs("exports", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"exports/wazuh_alerts_{timestamp}.csv"
    df_export[
        [
            "@timestamp",
            "agent.name",
            "rule.id",
            "rule.description",
            "rule.level",
            "rule.mitre.id",
            "data.win.eventdata.commandLine",
        ]
    ].to_csv(filename, index=False)

    st.success(f"Export terminé : {filename}")
    st.download_button(
        label="Télécharger le CSV",
        data=df_export.to_csv(index=False),
        file_name=f"wazuh_alerts_{timestamp}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Wazuh → Dataset ML")
    st.caption("Chargez vos alertes Wazuh, visualisez-les et exportez un dataset pour vos modèles.")

    today = date.today()
    default_start = today - timedelta(days=7)

    st.sidebar.header("Connexion")
    manager_url = st.sidebar.text_input("Wazuh Manager URL", value="https://192.168.100.11:55000")
    indexer_url = st.sidebar.text_input("Wazuh Indexer URL", value="https://192.168.100.11:9200")
    verify_tls = st.sidebar.checkbox("Vérifier TLS", value=False)

    manager_user = st.sidebar.text_input("Manager user", value=os.getenv("WAZUH_MANAGER_USER", ""))
    manager_pass = st.sidebar.text_input(
        "Manager password",
        value=os.getenv("WAZUH_MANAGER_PASS", ""),
        type="password",
    )
    indexer_user = st.sidebar.text_input("Indexer user", value=os.getenv("WAZUH_INDEXER_USER", ""))
    indexer_pass = st.sidebar.text_input(
        "Indexer password",
        value=os.getenv("WAZUH_INDEXER_PASS", ""),
        type="password",
    )

    st.sidebar.header("Filtres")
    date_range = st.sidebar.date_input(
        "Plage de dates",
        value=(default_start, today),
    )
    atomic_only = st.sidebar.checkbox("Atomic Red Team uniquement (rule.mitre.id)", value=False)

    available_agents = st.session_state.get("available_agents", [])
    available_mitre = st.session_state.get("available_mitre", [])
    available_rules = st.session_state.get("available_rules", [])
    available_levels = st.session_state.get("available_levels", [])

    selected_agents = st.sidebar.multiselect("agent.name", options=available_agents)
    selected_mitre = st.sidebar.multiselect("rule.mitre.id", options=available_mitre)
    selected_rules = st.sidebar.multiselect("rule.id", options=available_rules)
    selected_levels = st.sidebar.multiselect("rule.level", options=available_levels)

    test_button = st.sidebar.button("Tester connexion")
    load_button = st.sidebar.button("Charger données")
    export_button = st.sidebar.button("Exporter CSV")
    train_button = st.sidebar.button("Entraîner modèle severity")

    dt_range = build_datetime_range(date_range) or (
        datetime.combine(default_start, time.min),
        datetime.combine(today, time.max),
    )
    start_dt, end_dt = dt_range

    manager_cfg = {
        "base_url": manager_url,
        "username": manager_user,
        "password": manager_pass,
        "verify_tls": verify_tls,
    }
    indexer_cfg = {
        "base_url": indexer_url,
        "username": indexer_user,
        "password": indexer_pass,
        "verify_tls": verify_tls,
        "index_pattern": DEFAULT_INDEX_PATTERN,
    }

    if test_button:
        run_connection_tests(manager_cfg, indexer_cfg)

    df: Optional[pd.DataFrame] = None
    if load_button:
        if not indexer_user or not indexer_pass:
            st.error("Merci de renseigner les identifiants de l'indexer.")
        else:
            with st.spinner("Chargement des alertes (limité à 10k)..."):
                df = load_alerts(
                    base_url=indexer_url,
                    username=indexer_user,
                    password=indexer_pass,
                    verify_tls=verify_tls,
                    start=start_dt,
                    end=end_dt,
                    agents=selected_agents,
                    mitre_ids=selected_mitre,
                    rule_ids=selected_rules,
                    levels=selected_levels,
                    atomic_only=atomic_only,
                    limit=10_000,
                )
                st.success(f"{len(df)} alertes chargées")
                if not df.empty:
                    normalize_filters_from_df(df)

    if export_button:
        if not indexer_user or not indexer_pass:
            st.error("Merci de renseigner les identifiants de l'indexer pour exporter.")
        else:
            export_csv(
                indexer_cfg,
                start=start_dt,
                end=end_dt,
                agents=selected_agents,
                mitre_ids=selected_mitre,
                rule_ids=selected_rules,
                levels=selected_levels,
                atomic_only=atomic_only,
            )

    if df is not None:
        if df.empty:
            st.info("Aucune donnée à afficher.")
            return

        render_kpis(df)

        st.markdown("<div class='section-title'>Données filtrées</div>", unsafe_allow_html=True)
        display_cols = [
            "@timestamp",
            "agent.name",
            "rule.id",
            "rule.description",
            "rule.level",
            "severity",
            "rule.mitre.id_str",
            "data.win.eventdata.commandLine",
        ]
        existing_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[existing_cols], use_container_width=True, height=380)

        render_charts(df)

        st.markdown("<div class='section-title'>Modèle ML : classification de sévérité</div>", unsafe_allow_html=True)
        st.caption("Si la colonne 'severity' est absente, elle sera dérivée automatiquement depuis rule.level.")
        if train_button:
            progress = st.progress(0)
            status = st.empty()
            try:
                def report(msg: str, pct: float | None = None) -> None:
                    status.info(msg)
                    if pct is not None:
                        progress.progress(min(max(pct, 0.0), 1.0))

                result: TrainingArtifacts = train_severity_models(df, reporter=report)
                progress.progress(1.0)
                status.success(f"Modèle entraîné : {result.best_model_name}")

                st.write("Métriques (f1_macro, accuracy, balanced_accuracy) :")
                metrics_df = pd.DataFrame(result.metrics)
                st.dataframe(metrics_df, use_container_width=True)
                if not metrics_df.empty:
                    fig_metrics = px.bar(
                        metrics_df,
                        x="model",
                        y="f1_macro",
                        color="accuracy",
                        color_continuous_scale=[PRIMARY_COLOR, PRIMARY_COLOR],
                        title="Comparaison des modèles (F1 macro)",
                    )
                    fig_metrics.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_metrics, use_container_width=True)

                # Confusion matrix for best model
                cm_fig = ff.create_annotated_heatmap(
                    z=result.confusion_matrix,
                    x=result.class_labels,
                    y=result.class_labels,
                    colorscale=[[0, "#e8f5e9"], [1, PRIMARY_COLOR]],
                    showscale=True,
                )
                cm_fig.update_layout(title="Matrice de confusion (meilleur modèle)", template="plotly_white")
                st.plotly_chart(cm_fig, use_container_width=True)

                with st.expander("Classification report"):
                    st.text(result.classification_report)

                st.download_button(
                    "Télécharger le modèle (PKL)",
                    data=result.model_bytes,
                    file_name=result.model_path.name,
                    mime="application/octet-stream",
                )
                st.caption(f"Fichier sauvegardé dans {result.model_path} — métriques : {result.metrics_path}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur pendant l'entraînement : {exc}")
                progress.progress(0)
                status.error("Échec de l'entraînement.")


if __name__ == "__main__":
    main()
