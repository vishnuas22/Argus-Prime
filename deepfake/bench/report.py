"""Markdown report-card renderer for P0 baseline results."""


def _fmt(x):
    return f"`{x}`" if isinstance(x, float) else str(x)


SHORT = {"metadata": "metadata", "compression_history": "compression",
         "spectral_probe": "spectral", "real_distribution_probe": "realness",
         "perturbation_probe": "perturbation", "learned_aigc_probe": "learned"}


def render_report(results, n_real, n_fake, module_ids=None) -> str:
    module_ids = module_ids or ["metadata", "compression_history", "spectral_probe",
                                "real_distribution_probe", "perturbation_probe"]
    names = [SHORT.get(m, m) for m in module_ids]
    lines = [
        "# ARGUS_MASTER — P0 Baseline Report Card",
        "",
        f"Corpus: {n_real} real (COCO val2017 slice) vs {n_fake} fake (SD 1.5 local). "
        "Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).",
        "",
        "## Detection performance by laundering rung",
        "",
        "| Variant | Metric | " + " | ".join(names + ["FUSED"]) + " |",
        "|---|---|" + "---|" * (len(names) + 1),
    ]
    order = module_ids
    for vname, block in results.items():
        for metric in ("auc", "eer"):
            cells = []
            for m, n in zip(order, names):
                v = block.get(m, {}).get(metric)
                cells.append("—" if v is None else f"{v:.3f}")
            fv = block.get("fusion", {}).get(metric)
            cells.append("—" if fv is None else f"{fv:.3f}")
            lines.append(f"| {vname} | {metric.upper()} | " + " | ".join(cells) + " |")

    f0 = results.get("clean", {})
    lines += [
        "",
        "## Fusion behavior (clean rung)",
        "",
        f"- Brier(P(real)): {_fmt(f0.get('fusion', {}).get('brier_p_real'))}",
        f"- Abstention rate: {_fmt(f0.get('fusion', {}).get('abstention_rate'))}",
        f"- Mean trust score: {_fmt(f0.get('fusion', {}).get('mean_trust'))}",
        f"- Mean risk score: {_fmt(f0.get('fusion', {}).get('mean_risk'))}",
        f"- Evidence-capacity mix: {f0.get('evidence_capacity_mix')}",
        "",
        "## Runtime (ms/image, clean rung)",
        "",
        "| Stage | ms |",
        "|---|---|",
    ]
    rt = f0.get("runtime_ms", {})
    for stage, ms in rt.items():
        lines.append(f"| {stage} | {ms} |")
    lines += [
        "",
        "## Reading guide",
        "",
        "- AUC 0.5 = coin flip; higher is better. EER lower is better.",
        "- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.",
        "- Modules unavailable (missing envelope/reference data) show as —; "
        "build Argus data files before trusting those rows.",
    ]
    return "\n".join(lines) + "\n"
