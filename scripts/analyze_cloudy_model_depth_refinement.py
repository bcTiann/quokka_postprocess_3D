#!/usr/bin/env python3
"""Analyze a completed 37-depth scan; never launch or modify Cloudy models.

Both candidate interpolants use the SAME odd-index direct holdouts. The 10-node
axis uses indices 0,4,...,36; the 19-node axis uses 0,2,...,36. The additional
37-node grid itself has no independent midpoint validation in this analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LOG_DEPTH = np.linspace(-0.25, 2.0, 37)
LABELS = ("[C II] 158 micron", r"H$\alpha$", "H I 21 cm", "C III 977", "C III 1907",
          "C III 1909", "C IV 1548", "C IV 1551")
PLAIN_LABELS = ("[C II] 158 μm", "Hα", "H I 21 cm", "C III 977", "C III 1907",
                "C III 1909", "C IV 1548", "C IV 1551")
KEYS = ("cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909", "civ_1548", "civ_1551")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#555555")
COARSE_COLOR, FINE_COLOR = "#1564a0", "#d95f02"
TOUCH_EPS = 1.0e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_list(values: np.ndarray) -> list:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values)]


def serialized(values: np.ndarray) -> np.ndarray:
    """Cloudy maps serialize positive log coefficients to four decimal places."""
    result = values.copy()
    positive = np.isfinite(values) & (values > 0)
    result[positive] = 10.0 ** np.round(np.log10(values[positive]), 4)
    return result


def interpolate(nodes: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Two-endpoint lookup matching log-positive / linear-true-zero policy."""
    if np.any(query < nodes[0]) or np.any(query > nodes[-1]):
        raise ValueError("Cannot extrapolate model depth")
    upper = np.clip(np.searchsorted(nodes, query, side="right"), 1, len(nodes) - 1)
    lower = upper - 1
    fraction = ((query - nodes[lower]) / (nodes[upper] - nodes[lower]))[:, None]
    left, right = values[lower], values[upper]
    invalid = ((~np.isfinite(left)) & (1.0 - fraction > TOUCH_EPS)) | (
        (~np.isfinite(right)) & (fraction > TOUCH_EPS))
    zeros = ((left == 0) & (1.0 - fraction > TOUCH_EPS)) | ((right == 0) & (fraction > TOUCH_EPS))
    linear = np.nan_to_num(left, nan=0.0) * (1.0 - fraction) + np.nan_to_num(right, nan=0.0) * fraction
    log_left, log_right = np.zeros_like(left), np.zeros_like(right)
    np.log10(left, out=log_left, where=np.isfinite(left) & (left > 0))
    np.log10(right, out=log_right, where=np.isfinite(right) & (right > 0))
    result = np.where(zeros, linear, 10.0 ** (log_left * (1.0 - fraction) + log_right * fraction))
    result[invalid] = np.nan
    return result


def assemble(summary: dict) -> tuple[np.ndarray, dict, list]:
    values = np.full((7, 37, 8), np.nan)
    indexed, invalid = {}, []
    for record in summary["records"]:
        track = int(record["track"])
        if track not in range(7):
            raise ValueError("Unexpected track")
        matches = np.flatnonzero(np.abs(LOG_DEPTH - record["log_L_model_pc"]) < 1e-12)
        if len(matches) != 1:
            raise ValueError("Record depth is not on the 37-point axis")
        index = int(matches[0])
        if record.get("depth_index") != index:
            raise ValueError("Record depth index disagrees with its physical length")
        if record.get("origin") != ("new_direct" if index % 2 else "reused_pilot"):
            raise ValueError("Record origin disagrees with the reused/new holdout split")
        if (track, index) in indexed:
            raise ValueError("Duplicate track/depth record")
        indexed[track, index] = record
        checks = record.get("checks", {})
        coefficient = np.asarray(checks.get("emissivity_per_nH2", []), dtype=float)
        valid = checks.get("valid") is True and coefficient.shape == (8,) and np.isfinite(coefficient).all() and np.all(coefficient >= 0)
        if valid:
            values[track, index] = coefficient
        else:
            invalid.append(dict(track=track, depth_index=index, record_id=record.get("id"),
                                log_L_model_pc=float(LOG_DEPTH[index]),
                                kind="failed_record", issues=checks.get("issues", [])))
    for track in range(7):
        records = [record for (number, _), record in indexed.items() if number == track]
        if records:
            for key in ("log_nH", "log_T", "log_NH"):
                if any(not np.isclose(float(record[key]), float(records[0][key]), rtol=0, atol=1e-12) for record in records):
                    raise ValueError(f"Track {track} changes {key}; this is not a depth-only comparison")
        for index in range(37):
            if (track, index) not in indexed:
                invalid.append(dict(track=track, depth_index=index, log_L_model_pc=float(LOG_DEPTH[index]),
                                    kind="missing_record", issues=["No direct result was present in summary"]))
    return values, indexed, invalid


def verify_runner_comparisons(summary: dict, rows: list[dict]) -> dict:
    """Check the runner against independently assembled endpoint predictions."""
    checked = {}
    for key, prediction_key in (("comparisons_10", "coarse10"), ("comparisons_19", "fine19")):
        source = {(item["track"], item["depth_index"]): item for item in summary[key]}
        if len(source) != len(summary[key]) or set(source) != {
            (row["track"], row["depth_index"]) for row in rows
        }:
            raise ValueError(f"{key} does not use the same 126 unique new holdouts")
        count = 0
        for row in rows:
            original = source[row["track"], row["depth_index"]]
            direct = np.asarray(row["direct"], dtype=float)
            predicted = np.asarray(row[prediction_key], dtype=float)
            available = np.isfinite(direct).all() and np.isfinite(predicted).all()
            if original["valid"] != available:
                raise ValueError(f"{key} validity disagrees with direct/node evidence")
            if available:
                np.testing.assert_allclose(original["direct"], direct, rtol=1e-13, atol=0)
                np.testing.assert_allclose(original["interpolated"], predicted, rtol=1e-12, atol=0)
                count += 1
        checked[key] = dict(holdouts=126, independently_matched_available=count)
    return checked


def source_record_inventory(indexed: dict) -> list[dict]:
    """Tie displayed values back to each saved direct-calculation checkpoint."""
    inventory = []
    for (track, index), record in sorted(indexed.items()):
        source = record.get("source_record_path")
        digest = None
        if source is not None:
            path = Path(source)
            saved = json.loads(path.read_text())
            if saved.get("checks") != record.get("checks") or any(
                saved.get(name) != record.get(name)
                for name in ("log_nH", "log_T", "log_NH", "log_L_model_pc")
            ):
                raise ValueError(f"Summary differs from saved direct checkpoint: {path}")
            digest = sha256(path)
        inventory.append(dict(track=track, depth_index=index, origin=record["origin"],
                              source_record_path=source, source_record_sha256=digest))
    return inventory


def comparisons(values: np.ndarray) -> tuple[list, list, np.ndarray, np.ndarray]:
    holdouts = np.arange(1, 37, 2)
    direct = values[:, holdouts]
    predictions = []
    for step in (4, 2):
        nodes = np.arange(0, 37, step)
        predictions.append(np.array([interpolate(LOG_DEPTH[nodes], serialized(track[nodes]), LOG_DEPTH[holdouts])
                                     for track in values]))
    coarse, fine = predictions
    rows = []
    for track in range(7):
        for j, index in enumerate(holdouts):
            d, c, f = direct[track, j], coarse[track, j], fine[track, j]
            paired = np.isfinite(d) & np.isfinite(c) & np.isfinite(f)
            rows.append(dict(track=track, depth_index=int(index), log_L_model_pc=float(LOG_DEPTH[index]),
                             direct=_finite_list(d), coarse10=_finite_list(c), fine19=_finite_list(f),
                             paired_available=paired.tolist()))
    line_metrics = []
    for line in range(8):
        d, c, f = (array[:, :, line].ravel() for array in (direct, coarse, fine))
        paired = np.isfinite(d) & np.isfinite(c) & np.isfinite(f)
        # Dex statistics compare the identical positive subset for both models;
        # zero mismatches and zero/zero agreements remain separate, never floored.
        paired_positive = paired & (d > 0) & (c > 0) & (f > 0)
        eligible = np.flatnonzero(paired_positive)
        metrics = dict(line=KEYS[line], label=PLAIN_LABELS[line], total_new_holdouts=126,
                       paired_available=int(paired.sum()), unavailable=int((~paired).sum()),
                       paired_positive_for_dex=int(paired_positive.sum()),
                       direct_zero_count=int(np.sum(paired & (d == 0))),
                       direct_positive_count=int(np.sum(paired & (d > 0))))
        for label, p in (("coarse10", c), ("fine19", f)):
            dex = np.abs(np.log10(p[paired_positive] / d[paired_positive]))
            ratio_mask = paired & (d > 0)
            relative = np.abs(p[ratio_mask] / d[ratio_mask] - 1)
            local = dict(zero_mismatch_count=int(np.sum(paired & ((p == 0) != (d == 0)))),
                         zero_agreement_count=int(np.sum(paired & (p == 0) & (d == 0))),
                         max_absolute_coefficient_error=float(np.max(np.abs(p[paired] - d[paired]))) if paired.any() else None,
                         max_relative_error=float(relative.max()) if relative.size else None,
                         median_absolute_dex=float(np.median(dex)) if dex.size else None,
                         p90_absolute_dex=float(np.quantile(dex, .90)) if dex.size else None,
                         p95_absolute_dex=float(np.quantile(dex, .95)) if dex.size else None,
                         max_absolute_dex=float(dex.max()) if dex.size else None)
            if dex.size:
                flat = int(eligible[np.argmax(dex)])
                track, position = divmod(flat, 18)
                index = int(holdouts[position])
                local["worst"] = dict(track=track, depth_index=index, L_pc=float(10.0**LOG_DEPTH[index]),
                                      direct=float(d[flat]), interpolated=float(p[flat]),
                                      interpolated_over_direct=float(p[flat]/d[flat]))
            else:
                local["worst"] = None
            metrics[label] = local
        line_metrics.append(metrics)
    return rows, line_metrics, coarse, fine


def save_figure(fig, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def _plot_nonpositive_strip(ax, lengths, values, color, *, label=False, offset=0.0):
    ax.scatter(lengths[np.isfinite(values) & (values == 0)],
               np.full(np.sum(np.isfinite(values) & (values == 0)), offset),
               color=color, marker="v", s=10, clip_on=False)
    ax.scatter(lengths[~np.isfinite(values)], np.full(np.sum(~np.isfinite(values)), 1.0 + offset),
               color=color, marker="x", s=10, clip_on=False)
    ax.set_ylim(-.6, 1.6)
    ax.set_yticks((0, 1), ("zero", "missing / failed") if label else ("", ""), fontsize=7)
    ax.set_xscale("log")
    ax.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_tracks(values, indexed, output):
    fig = plt.figure(figsize=(13.2, 14.0))
    outer = fig.add_gridspec(4, 2, hspace=.37, wspace=.24)
    lengths = 10.0**LOG_DEPTH
    for line in range(8):
        inner = outer[line // 2, line % 2].subgridspec(2, 1, height_ratios=(1, .15), hspace=.02)
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1], sharex=ax)
        for track in range(7):
            coefficient = values[track, :, line]
            ax.plot(lengths, np.where(coefficient > 0, coefficient, np.nan), "o-",
                    color=COLORS[track], ms=2.7, lw=1, alpha=.9)
            _plot_nonpositive_strip(strip, lengths, coefficient, COLORS[track], label=True,
                                    offset=(track - 3) * .11)
        ax.set(xscale="log", yscale="log", title=LABELS[line])
        ax.tick_params(labelbottom=False)
        ax.set_ylabel(r"$\epsilon/n_{\rm H}^{2}$ [erg s$^{-1}$ cm$^3$]", fontsize=9)
        strip.set_xlabel("Model thickness L [pc]", fontsize=9)
    legends = []
    for track in range(7):
        record = next((record for (number, _), record in indexed.items() if number == track), None)
        text = f"Track {track}"
        if record:
            text += f": nH={10.0**record['log_nH']:.3g}, T={10.0**record['log_T']:.4g} K, log NH={record['log_NH']:g}"
        legends.append(Line2D([], [], color=COLORS[track], marker="o", ms=4, lw=1, label=text))
    fig.legend(handles=legends, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(.5, .005))
    fig.suptitle("37 direct depths per fixed-state track — eight emission coefficients", fontsize=14, y=.985)
    fig.text(.5, .958, "Lower strips: true zeros / unavailable states, offset by track for visibility; no artificial positive floor.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.925, bottom=.145)
    save_figure(fig, output, "all_tracks_eight_lines")


def plot_errors(values, coarse, fine, output):
    direct = values[:, 1::2]
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for line, ax in enumerate(axes.ravel()):
        d, c, f = (array[:, :, line].ravel() for array in (direct, coarse, fine))
        paired = np.isfinite(d) & np.isfinite(c) & np.isfinite(f) & (d > 0) & (c > 0) & (f > 0)
        for prediction, color, label in ((c, COARSE_COLOR, "10 nodes (0.25 dex)"), (f, FINE_COLOR, "19 nodes (0.125 dex)")):
            error = np.sort(np.abs(np.log10(prediction[paired] / d[paired])))
            if error.size:
                ax.step(error, np.arange(1, len(error)+1)/len(error), where="post", color=color, label=label)
        ax.set(xlabel="Absolute log10(interpolated / direct) [dex]", ylabel="Fraction of paired positive checks",
               title=f"{LABELS[line]}: {np.sum(paired)}/126 common positive checks", ylim=(0, 1.03))
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Fair comparison: both interpolants evaluated at the same 126 new direct holdouts", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_figure(fig, output, "paired_holdout_error_cdf")


def plot_worst(values, indexed, metrics, output):
    fig = plt.figure(figsize=(12.4, 8))
    outer = fig.add_gridspec(1, 2, wspace=.25)
    lengths = 10.0**LOG_DEPTH
    holdouts = np.arange(1, 37, 2)
    query = np.linspace(LOG_DEPTH[0], LOG_DEPTH[-1], 721)
    for column, line in enumerate((0, 2)):
        worst = metrics[line]["fine19"]["worst"] or metrics[line]["coarse10"]["worst"]
        if worst is None:
            ax = fig.add_subplot(outer[column])
            ax.text(.5, .5, "No paired positive holdouts", transform=ax.transAxes, ha="center")
            continue
        track = worst["track"]
        inner = outer[column].subgridspec(3, 1, height_ratios=(3.2, .32, 1.65), hspace=.11)
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1], sharex=ax)
        error_ax = fig.add_subplot(inner[2], sharex=ax)
        direct = values[track, :, line]
        for step, color, label in ((4, COARSE_COLOR, "10-node interpolant"), (2, FINE_COLOR, "19-node interpolant")):
            nodes = np.arange(0, 37, step)
            curve = interpolate(LOG_DEPTH[nodes], serialized(values[track, nodes]), query)[:, line]
            predicted = interpolate(LOG_DEPTH[nodes], serialized(values[track, nodes]), LOG_DEPTH[holdouts])[:, line]
            ax.plot(10.0**query, np.where(curve > 0, curve, np.nan), color=color, lw=1.2, label=label)
            good = np.isfinite(predicted) & np.isfinite(direct[holdouts]) & (predicted > 0) & (direct[holdouts] > 0)
            error = np.full(len(holdouts), np.nan)
            error[good] = np.log10(predicted[good] / direct[holdouts][good])
            error_ax.plot(lengths[holdouts], error, "o-", color=color, ms=3, lw=.8)
        ax.plot(lengths, np.where(direct > 0, direct, np.nan), "o", color="black", ms=4,
                label=f"Direct values ({np.isfinite(direct).sum()}/37 valid)")
        ax.scatter(lengths[holdouts], np.where(direct[holdouts] > 0, direct[holdouts], np.nan),
                   facecolors="none", edgecolors="#009E73", s=44, label="New holdouts (18 per track)")
        _plot_nonpositive_strip(strip, lengths, direct, "black", label=True)
        ax.axvline(worst["L_pc"], color="gray", ls=":", lw=1)
        error_ax.axvline(worst["L_pc"], color="gray", ls=":", lw=1)
        error_ax.axhline(0, color="black", lw=.7)
        record = next(record for (number, _), record in indexed.items() if number == track)
        ax.set(xscale="log", yscale="log", title=f"{LABELS[line]}, track {track}",
               ylabel=r"$\epsilon/n_{\rm H}^{2}$ [erg s$^{-1}$ cm$^3$]")
        ax.text(.03, .03, f"nH={10.0**record['log_nH']:.4g} cm$^{{-3}}$\nT={10.0**record['log_T']:.4g} K; log NH={record['log_NH']:g}",
                transform=ax.transAxes, fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(labelbottom=False)
        strip.tick_params(labelbottom=False)
        error_ax.set(xscale="log", xlabel="Model thickness L [pc]", ylabel="log10(interpolated / direct)")
    fig.suptitle("Worst 19-node positive holdout for C II and H I — direct 37-point tracks", fontsize=12)
    fig.subplots_adjust(top=.92, bottom=.1)
    save_figure(fig, output, "worst_cii_hi_depth_tracks")


def _fmt(value, precision=5):
    return "不可定义 / 无有效值" if value is None else f"{value:.{precision}g}"


def write_readme(output, report, indexed):
    counts = report["counts"]
    cii, halpha, hi = report["per_line"][:3]
    text = f'''# Cloudy 长度轴加密：10 点与 19 点的同点比较

本分析读取 `../summary.json`，没有启动 Cloudy，也没有修改物理设置或 H I 的 zone 控制。原始摘要与分析脚本的 SHA256 见 `diagnostic.json`。

37 点扫描覆盖 7 条固定 `(nH,T,NH)` 轨迹，每条长度范围为 0.562341–100 pc。轴上共有 259 个位置；当前记录 {counts['records']} 个（复用 {counts['reused_records']} 个，新增 {counts['new_records']} 个），通过记录中输入、几何和收敛检查且具有完整非负八线输出的状态为 **{counts['valid_records']} 个**。失败记录 {counts['failed_records']} 个，缺失记录 {counts['missing_records']} 个，单独保留；不会把失败当成零发射率。

在同一批新增检查点上，C II 的最大绝对 dex 误差为 **{_fmt(cii['coarse10']['max_absolute_dex'])} → {_fmt(cii['fine19']['max_absolute_dex'])}**，H I 21 cm 为 **{_fmt(hi['coarse10']['max_absolute_dex'])} → {_fmt(hi['fine19']['max_absolute_dex'])}**，Hα 为 **{_fmt(halpha['coarse10']['max_absolute_dex'])} → {_fmt(halpha['fine19']['max_absolute_dex'])}**。箭头表示 10 点表到 19 点表；加密后各条线、各项误差指标的变化并不一致，不能把缩小长度步长直接当成所有谱线都更准确的保证。

## 比较如何保持公平

- 10 点表采用每 0.25 dex 的位置（37 点轴的索引 0,4,…,36）。
- 19 点表采用每 0.125 dex 的位置（索引 0,2,…,36）；它吸收了先前 10 点测试的中点作为节点。
- **两种表均只在同一批 126 个新增位置比较**（每条轨迹索引 1,3,…,35）。这些点不属于任一种表的节点，没有把旧中点上的误差与新四分位点上的误差混为一谈。
- 节点正系数先按实际 map 规则将 log10 值舍入到四位小数；正值两端采用 log 系数插值，有真实零值参与时改用线性系数插值。直接检查值保持原始输出精度。
- dex 误差只比较两种预测与直接值都为正的共同子集。零/零一致、零/非零不一致及不可用检查单独计数，不给零值添加人为底值。

分析器由原始端点独立重算两种预测，并逐点与运行程序保存的比较数组核对；核对结果保存在 `diagnostic.json`。

## 八条线：同一批检查点上的误差

下表为绝对 `log10(插值值/直接值)`，单位 dex。中位数、90% 分位与最大值使用同一正值子集；它们没有按 simulation cell 数量、质量或体积加权。95% 分位也保存在 `diagnostic.json`。

| 谱线 | 共同可用 /126 | 共同正值 /126 | 10 点：中位 /90% /最大 dex | 19 点：中位 /90% /最大 dex | 零不一致：10/19 点 |
|---|---:|---:|---|---|---:|
'''
    for metric in report["per_line"]:
        coarse, fine = metric["coarse10"], metric["fine19"]
        def cells(value):
            return " / ".join(_fmt(value[key]) for key in ("median_absolute_dex", "p90_absolute_dex", "max_absolute_dex"))
        text += f"| {metric['label']} | {metric['paired_available']} | {metric['paired_positive_for_dex']} | {cells(coarse)} | {cells(fine)} | {coarse['zero_mismatch_count']}/{fine['zero_mismatch_count']} |\n"
    zero_counts = [f"{metric['label']}：{metric['direct_zero_count']}" for metric in report["per_line"]
                   if metric["direct_zero_count"]]
    text += "\n共同可用检查中的真实直接零值数量：" + ("；".join(zero_counts) if zero_counts else "各条线均为 0") + "。这些点没有被当成失败，也没有被混入正值 dex 分位数。\n"
    text += "\n## C II 与 H I 的最差 19 点检查\n\n"
    for line in (0, 2):
        metric = report["per_line"][line]
        for grid in ("coarse10", "fine19"):
            worst = metric[grid]["worst"]
            if worst:
                text += (f"- {metric['label']}，{'10 点' if grid == 'coarse10' else '19 点'}：轨迹 {worst['track']}，"
                         f"L={worst['L_pc']:.7g} pc，最大绝对误差 {metric[grid]['max_absolute_dex']:.6g} dex，"
                         f"插值/直接值={worst['interpolated_over_direct']:.7g}。\n")
    text += '''
这次只改变长度采样。曲线中若保留快速转变或不规则变化，当前图与检查只能显示这种长度依赖，不能单独证明其物理或数值成因，也不能证明 H I 已对 zone 分区收敛。

## 图与数据

- `all_tracks_eight_lines.png/.pdf`：八个谱线面板，每个面板显示七条轨迹的 37 个直接计算位置。下方单独条带显示真实零值与缺失/失败位置，颜色按轨迹区分并稍作上下错开以免相互遮挡；错开不代表不同物理值。主图没有使用假正值替代它们。
- `paired_holdout_error_cdf.png/.pdf`：两种表在相同新增检查点上的误差累积分布。横轴在接近零处使用线性区间，其他区域使用对称对数显示。
- `worst_cii_hi_depth_tracks.png/.pdf`：按 19 点表的最大正值 dex 误差选择 C II 与 H I 的轨迹，显示 37 点直接输出、10/19 点预测与独立检查误差。
- `diagnostic.json`：每条线统计、每个新增检查点的直接值与两种预测、失败/缺失位置及来源哈希。

## 结论的边界

这组新增直接计算能够检验 19 点候选长度轴在七条代表轨迹上的误差；是否足够仍须结合明确的误差要求决定。**37 个直接位置画出了更细曲线，但没有在 0.0625 dex 间隔的独立中点（0.03125 dex 偏移）上检验，不能据此称 37 点表已通过精度验证。**本结果也不等于整个四维表的误差验证，不提供全 snapshot 的光度影响估计，没有采用新的正式表。
'''
    (output / "README.md").write_text(text)


def analyze(summary_path: Path, output: Path) -> dict:
    summary_path = summary_path.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    values, indexed, invalid = assemble(summary)
    rows, metrics, coarse, fine = comparisons(values)
    runner_agreement = verify_runner_comparisons(summary, rows)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="completed_analysis" if summary.get("status") == "completed" else "partial_analysis",
                  summary_path=str(summary_path), summary_sha256=sha256(summary_path),
                  analyzer_sha256=sha256(Path(__file__)),
                  counts=dict(expected_records=259, records=len(indexed),
                              valid_records=int(np.isfinite(values).all(axis=2).sum()),
                              reused_records=sum(record["origin"] == "reused_pilot" for record in indexed.values()),
                              new_records=sum(record["origin"] == "new_direct" for record in indexed.values()),
                              failed_records=sum(item["kind"] == "failed_record" for item in invalid),
                              missing_records=sum(item["kind"] == "missing_record" for item in invalid),
                              shared_new_holdout_positions=126,
                              invalid_new_holdout_positions=sum(item["depth_index"] % 2 == 1 for item in invalid)),
                  logarithmic_depth_axis_pc=LOG_DEPTH.tolist(),
                  coarse10_node_indices=list(range(0, 37, 4)), fine19_node_indices=list(range(0, 37, 2)),
                  shared_holdout_indices=list(range(1, 37, 2)), invalid_records=invalid,
                  runner_comparison_crosscheck=runner_agreement,
                  source_records=source_record_inventory(indexed),
                  per_line=metrics, comparisons=rows,
                  limitations=["No independent validation of a 37-node interpolant.",
                               "No full-snapshot luminosity or cell/mass-weighted error estimate.",
                               "No changes to Cloudy zone controls or physical inputs."])
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    plot_tracks(values, indexed, output)
    plot_errors(values, coarse, fine, output)
    plot_worst(values, indexed, metrics, output)
    write_readme(output, report, indexed)
    return {"status": report["status"], "output": str(output), "counts": report["counts"],
            "per_line": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.summary, args.output_dir or args.summary.parent / "analysis"), indent=2))


if __name__ == "__main__":
    main()
