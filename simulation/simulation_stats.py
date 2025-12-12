import queue
from collections import defaultdict
from enum import IntEnum, Enum
import warnings
import numpy as np
import pandas as pd
import itertools
from lifelines import CoxPHFitter
from pyexcelerate import Workbook
from scipy.interpolate import interp1d

import sirakaya
from tqdm.auto import tqdm as _tqdm


def summarize_trajectory(
    self,
    p_freeze,
    state_lst_columns=["state_lst"],
    state_columns=["state"],
    windowsize=10,
):
    snaps = len(self.t_dfi)
    results = []
    results_df = []
    results_flow = []
    results_retention = []
    error_accumulated = 0.0
    pbar = _tqdm(range(0, snaps - 1), desc="Summarizing trajectory")

    # compute the (state, t) trajectory
    for p in pbar:
        df_begin = (
            self.t_dfi[p]
            .copy()
            .dropna(subset=["state", "state_lst"])
            .assign(
                state=lambda df: df["state"].apply(lambda x: x.value),
                state_lst=lambda df: df["state_lst"].apply(lambda x: x.value),
            )
        )
        lst_cols_avail = [c for c in state_lst_columns if c in df_begin.columns]
        cur_cols_avail = [c for c in state_columns if c in df_begin.columns]

        grp_keys_lst = lst_cols_avail
        grp_keys_cur = cur_cols_avail
        grp_keys_flow = grp_keys_lst + cur_cols_avail
        _boundary = p + p_freeze
        _present_start_mask = (df_begin["ep_arrival"] < _boundary) & (
            df_begin["ep_leaving"] >= _boundary
        )
        _present_end_mask = (df_begin["ep_arrival"] <= _boundary) & (
            df_begin["ep_leaving"] > _boundary
        )
        _leave_now_mask = df_begin["ep_leaving"] == _boundary
        _arrive_now_mask = df_begin["ep_arrival"] == _boundary
        _recid_now_mask = df_begin["ep_lastre"] == _boundary

        df_presence_begin = df_begin.loc[_present_start_mask]
        x0 = df_presence_begin.groupby(grp_keys_lst)["index"].count()
        x = df_begin.loc[_present_end_mask].groupby(grp_keys_cur)["index"].count()
        y = df_begin.loc[_recid_now_mask].groupby(grp_keys_flow)["index"].count()
        yout = y.groupby(grp_keys_lst).sum()
        yin = y.groupby(grp_keys_cur).sum()
        lmd = df_begin.loc[_arrive_now_mask].groupby(grp_keys_lst)["index"].count()
        lv = (
            df_presence_begin.loc[_leave_now_mask]
            .groupby(grp_keys_lst)["index"]
            .count()
        )
        tau = df_presence_begin.groupby(grp_keys_lst)["bool_treat"].sum()
        df_traj = (
            pd.DataFrame(
                {
                    "x0": x0,
                    "x": x,
                    "lmd": lmd,
                    "lv": lv,
                    "yin": yin,
                    "yout": yout,
                    "tau": tau,
                }
            )
            .fillna(0)
            .rename_axis(index=grp_keys_lst)
            .assign(
                tau_rel=lambda df: df["tau"] / (df["x0"] + 1e-3),
                xcal=lambda df: df.apply(
                    lambda row: row["x0"]
                    + row["lmd"]
                    - row["lv"]
                    + row["yin"]
                    - row["yout"],
                    axis=1,
                ),
                error=lambda df: df["xcal"] - df["x"],
            )
        )
        error = df_traj["error"].abs().max()
        error_accumulated += error
        pbar.set_postfix(error_max_average=f"{error_accumulated/(p+1):.2f}")

        rr = {
            "y": y.to_dict() if len(y) > 0 else {},
            **df_traj.to_dict(),
        }
        results.append(rr)
        results_df.append(df_traj)

        # compute exact flows from previous state(s) to current state(s)
        try:
            # classify x/y strictly by arrest count change at the boundary:
            # y: recidivism occurs at boundary; x: present and no recidivism and not leaving
            # x: present at start and no recid at boundary (ignoring leave/end presence)
            x_flow_df = (
                df_begin.loc[_present_start_mask & (~_recid_now_mask)]
                .groupby(grp_keys_flow)["index"]
                .count()
                .rename("count")
                .reset_index()
            )
            x_flow_df["source"] = "x"
            # ratio per origin (code_county + previous-state columns) within source x
            _xtot = x_flow_df.groupby(grp_keys_lst)["count"].transform("sum")
            x_flow_df["ratio"] = np.where(_xtot > 0, x_flow_df["count"] / _xtot, 0.0)

            # y: present at start and recid at boundary (ignoring leave/end presence)
            y_flow_df = (
                df_begin.loc[_present_start_mask & _recid_now_mask]
                .groupby(grp_keys_flow)["index"]
                .count()
                .rename("count")
                .reset_index()
            )
            y_flow_df["source"] = "y"
            # ratio per origin (code_county + previous-state columns) within source y
            _ytot = y_flow_df.groupby(grp_keys_lst)["count"].transform("sum")
            y_flow_df["ratio"] = np.where(_ytot > 0, y_flow_df["count"] / _ytot, 0.0)

            # arrivals as separate source with previous-state marked 'NEW'
            arrival_flow_df = (
                df_begin.loc[_arrive_now_mask]
                .groupby(grp_keys_cur)["index"]
                .count()
                .rename("count")
                .reset_index()
            )
            for prev_col in lst_cols_avail:
                arrival_flow_df[prev_col] = "NEW"
            _missing_cols = [
                c for c in grp_keys_flow if c not in arrival_flow_df.columns
            ]
            for c in _missing_cols:
                if c in lst_cols_avail:
                    arrival_flow_df[c] = "NEW"
                else:
                    arrival_flow_df[c] = np.nan
            arrival_flow_df = arrival_flow_df[grp_keys_flow + ["count"]]
            arrival_flow_df["source"] = "new"
            _ntot_new = arrival_flow_df.groupby(grp_keys_lst)["count"].transform("sum")
            arrival_flow_df["ratio"] = np.where(
                _ntot_new > 0, arrival_flow_df["count"] / _ntot_new, 0.0
            )

            flow_df = pd.concat(
                [x_flow_df, y_flow_df, arrival_flow_df], ignore_index=True
            ).assign(
                state_key=lambda df: df["state"].apply(
                    self.state_defs.state_key_range.get
                ),
                state_lst_key=lambda df: df["state_lst"].apply(
                    self.state_defs.state_key_range.get
                ),
            )
        except Exception:
            # fallback to empty frame if grouping keys are missing
            flow_df = pd.DataFrame(columns=grp_keys_flow + ["count", "source", "ratio", "ratio_all"])  # type: ignore

        # compute ratios: within-source and across all sources per origin
        if len(flow_df) > 0:
            _src_tot = flow_df.groupby(grp_keys_lst + ["source"], group_keys=False)[
                "count"
            ].transform("sum")
            flow_df["ratio"] = np.where(_src_tot > 0, flow_df["count"] / _src_tot, 0.0)
            _orig_tot = flow_df.groupby(grp_keys_lst, group_keys=False)[
                "count"
            ].transform("sum")
            flow_df["ratio_all"] = np.where(
                _orig_tot > 0, flow_df["count"] / _orig_tot, 0.0
            )

        # sort by source, previous-state columns, then current-state columns
        sort_cols = ["source"] + lst_cols_avail + cur_cols_avail
        sort_cols = [c for c in sort_cols if c in flow_df.columns]
        if len(sort_cols) > 0:
            flow_df = flow_df.sort_values(by=sort_cols).reset_index(drop=True)

        flow_df["snap"] = p
        results_flow.append(flow_df)

        # compute retention per origin based on starts (including arrivals) and leaves
        _start_mask_union = _present_start_mask | _arrive_now_mask
        _starts = df_begin.loc[_start_mask_union].groupby(grp_keys_lst)["index"].count()
        _leaves = (
            df_begin.loc[_leave_now_mask & _start_mask_union]
            .groupby(grp_keys_lst)["index"]
            .count()
        )
        _ret_df = (
            pd.DataFrame({"total": _starts, "left": _leaves})
            .fillna(0)
            .assign(
                stay=lambda d: d["total"] - d["left"],
                retention_ratio=lambda d: np.where(
                    d["total"] > 0, (d["total"] - d["left"]) / d["total"], 0.0
                ),
            )
            .reset_index()
        )
        _ret_df["snap"] = p
        retention_df = _ret_df.assign(
            state_lst_key=lambda df: df["state_lst"].apply(
                self.state_defs.state_key_range.get
            ),
        )

        results_retention.append(retention_df)

    # compute moving-average counts first, then ratios over last N episodes
    if len(results_flow) > 0:
        _all_flow = pd.concat(results_flow, ignore_index=True)
        needed_cols = grp_keys_flow + ["source", "snap", "count"]
        if all(c in _all_flow.columns for c in needed_cols):
            _all_flow = _all_flow.sort_values(by=grp_keys_flow + ["source", "snap"])
            # rolling average counts per origin+source+destination
            _all_flow["count_ma"] = (
                _all_flow.groupby(grp_keys_flow + ["source"], group_keys=False)["count"]
                .rolling(window=windowsize, min_periods=10)
                .mean()
                .reset_index(level=list(range(len(grp_keys_flow) + 1)), drop=True)
            )
            # per-source normalization across destinations
            _all_flow["origin_source_total_ma"] = _all_flow.groupby(
                grp_keys_lst + ["source", "snap"], group_keys=False
            )["count_ma"].transform("sum")
            _all_flow["ratio_ma"] = np.where(
                _all_flow["origin_source_total_ma"] > 0,
                _all_flow["count_ma"] / _all_flow["origin_source_total_ma"],
                0.0,
            )
            # across all sources normalization
            _all_flow["origin_total_ma"] = _all_flow.groupby(
                grp_keys_lst + ["snap"], group_keys=False
            )["count_ma"].transform("sum")
            _all_flow["ratio_all_ma"] = np.where(
                _all_flow["origin_total_ma"] > 0,
                _all_flow["count_ma"] / _all_flow["origin_total_ma"],
                0.0,
            )
            results_flow = [
                _all_flow[_all_flow["snap"] == snap].reset_index(drop=True)
                for snap in sorted(_all_flow["snap"].unique())
            ]

    # compute retention
    try:
        if len(results_retention) > 0:
            _all_ret = pd.concat(results_retention, ignore_index=True)
            if all(
                c in _all_ret.columns for c in grp_keys_lst + ["snap", "stay", "left"]
            ):
                _all_ret = _all_ret.sort_values(by=grp_keys_lst + ["snap"])
                # rolling average counts then compute ratio
                _all_ret["stay_ma"] = (
                    _all_ret.groupby(grp_keys_lst, group_keys=False)["stay"]
                    .rolling(window=windowsize, min_periods=10)
                    .mean()
                    .reset_index(level=list(range(len(grp_keys_lst))), drop=True)
                )
                _all_ret["left_ma"] = (
                    _all_ret.groupby(grp_keys_lst, group_keys=False)["left"]
                    .rolling(window=windowsize, min_periods=10)
                    .mean()
                    .reset_index(level=list(range(len(grp_keys_lst))), drop=True)
                )
                _all_ret["total_ma"] = _all_ret["stay_ma"] + _all_ret["left_ma"]
                _all_ret["retention_ratio_ma"] = np.where(
                    _all_ret["total_ma"] > 0,
                    _all_ret["stay_ma"] / _all_ret["total_ma"],
                    0.0,
                )
                results_retention = [
                    _all_ret[_all_ret["snap"] == snap].reset_index(drop=True)
                    for snap in sorted(_all_ret["snap"].unique())
                ]
                results_retention = [
                    _all_ret[_all_ret["snap"] == snap].reset_index(drop=True)
                    for snap in sorted(_all_ret["snap"].unique())
                ]
    except Exception:
        pass

    # sum of last 20 results
    # without having na
    from functools import reduce

    mean_df = reduce(
        lambda x, y: x.add(y, fill_value=0), results_df[-windowsize:]
    ).astype(float)
    mean_df = mean_df / windowsize

    # build transition matrices Px and Py (state_lst_key -> state_key)

    cols_needed = {
        "state_key",
        "state_lst_key",
        "source",
        "snap",
        "ratio_ma",
    }
    if cols_needed.issubset(_all_flow.columns):
        num_states = max(self.state_defs.state_key_range.values()) + 1
        Px = np.zeros((num_states, num_states), dtype=float)
        Py = np.zeros((num_states, num_states), dtype=float)
        last_snap = int(sorted(_all_flow["snap"].unique())[-1])
        flow_last = _all_flow[_all_flow["snap"] == last_snap]

        def _fill_matrix(df_src, M):
            dfw = df_src.copy().fillna(0.0)
            dfw = dfw.dropna(subset=["state_key", "state_lst_key"])  # ensure valid keys
            if len(dfw) == 0:
                return
            # use ratio_ma directly (already conditional per origin),
            # average across counties and renormalize per origin
            grouped = dfw.groupby(["state_lst_key", "state_key"], as_index=False)[
                "ratio_ma"
            ].mean()
            row_sum = grouped.groupby("state_lst_key")["ratio_ma"].sum().to_dict()
            for _, r in grouped.iterrows():
                i = int(r["state_lst_key"])  # origin
                j = int(r["state_key"])  # destination
                s = float(row_sum.get(i, 0.0))
                p = (float(r["ratio_ma"]) / s) if s > 0 else 0.0
                if 0 <= i < num_states and 0 <= j < num_states:
                    M[i, j] = p

        _fill_matrix(flow_last[flow_last["source"] == "x"], Px)
        _fill_matrix(flow_last[flow_last["source"] == "y"], Py)

        # save on simulator for downstream use
        self.Px = Px
        self.Py = Py
        # also save flattened vectors; fill zeros for missing entries
        self.Px_vec = Px.flatten()
        self.Py_vec = Py.flatten()

    # also, save x, x0, lmd, lv, yout, yin, tau from mean_df
    # retention_ratio_ma from results_retention[-1]

    try:
        # map mean_df to include state_lst_key for vector placement
        _md = mean_df.reset_index()
        if "state_lst" in _md.columns:
            _md["state_lst_key"] = _md["state_lst"].map(self.state_defs.state_key_range)
        num_states = max(self.state_defs.state_key_range.values()) + 1

        def _to_vec(column_name: str) -> np.ndarray:
            vec = np.zeros(num_states, dtype=float)
            if column_name in _md.columns and "state_lst_key" in _md.columns:
                for _, r in _md.dropna(subset=["state_lst_key"]).iterrows():
                    k = (
                        int(r["state_lst_key"])
                        if not pd.isna(r["state_lst_key"])
                        else -1
                    )
                    if 0 <= k < num_states:
                        vec[k] = (
                            float(r[column_name])
                            if not pd.isna(r[column_name])
                            else 0.0
                        )
            return vec

        self.x0_vec = _to_vec("x0")
        self.x_vec = _to_vec("x")
        self.lmd_vec = _to_vec("lmd")
        self.lv_vec = _to_vec("lv")
        self.yout_vec = _to_vec("yout")
        self.yin_vec = _to_vec("yin")
        self.tau_vec = _to_vec("tau")
    except Exception:
        pass

    try:
        # retention vector from latest retention snapshot
        if len(results_retention) > 0:
            last_ret = results_retention[-1]
            num_states = max(self.state_defs.state_key_range.values()) + 1
            self.retention_vec = np.zeros(num_states, dtype=float)
            col = (
                "retention_ratio_ma"
                if "retention_ratio_ma" in last_ret.columns
                else "retention_ratio"
            )
            if "state_lst_key" in last_ret.columns and col in last_ret.columns:
                for _, r in last_ret.dropna(subset=["state_lst_key"]).iterrows():
                    k = (
                        int(r["state_lst_key"])
                        if not pd.isna(r["state_lst_key"])
                        else -1
                    )
                    if 0 <= k < num_states:
                        self.retention_vec[k] = max(
                            float(r[col]) if not pd.isna(r[col]) else 0.0, 0.1
                        )
    except Exception:
        pass

    return (
        mean_df.reset_index().assign(
            state_lst_key=lambda df: df["state_lst"].map(
                self.state_defs.state_key_range
            ),
        ),
        results,
        results_df,
        results_flow,
        results_retention,
    )


def evaluation_metrics(results_df):
    """
    Compute evaluation metrics from trajectory results.

    Each result in results_df is a dataframe with columns:
        x0, x, lmd, lv, yin, yout, tau, tau_rel, xcal, error

    Example:
        state_lst     x0    x  lmd   lv  yin  yout  tau   tau_rel  xcal  error
    (25.0, 4.0, 1.0)  1.0  1.0  0.0  0.0  1.0   1.0  1.0  0.999001   1.0    0.0
    (19.0, 3.0, 1.0)  1.0  1.0  0.0  0.0  1.0   1.0  1.0  0.999001   1.0    0.0
    (44.0, 6.0, 1.0)  1.0  0.0  0.0  0.0  0.0   1.0  1.0  0.999001   0.0    0.0
    (20.0, 6.0, 1.0)  1.0  1.0  0.0  0.0  0.0   0.0  1.0  0.999001   1.0    0.0
    (22.0, 5.0, 1.0)  2.0  1.0  0.0  0.0  0.0   1.0  1.0  0.499750   1.0    0.0
    (13.0, 3.0, 1.0)  3.0  0.0  1.0  0.0  0.0   3.0  1.0  0.333222   1.0    1.0
    (15.0, 3.0, 1.0)  3.0  0.0  0.0  0.0  0.0   2.0  1.0  0.333222   1.0    1.0
    (5.0, 6.0, 1.0)   4.0  3.0  0.0  0.0  1.0   1.0  1.0  0.249938   4.0    1.0
    (14.0, 3.0, 1.0)  4.0  6.0  0.0  0.0  3.0   1.0  1.0  0.249938   6.0    0.0
    (7.0, 5.0, 1.0)   4.0  5.0  1.0  0.0  3.0   2.0  1.0  0.249938   6.0    1.0

    Returns dict with vectors (one value per episode):
        - total_population: total individuals present (sum of x0)
        - total_offenses: total recidivism events (sum of yin)
        - total_enrollment: total treated individuals (sum of tau)
        - total_arrivals: new arrivals (sum of lmd)
        - total_departures: left system (sum of lv)
        - offense_rate: offenses / population
        - enrollment_rate: enrolled / population
    """
    metrics = {
        "total_population": [],
        "total_offenses": [],
        "total_enrollment": [],
        "total_arrivals": [],
        "total_departures": [],
        "offense_rate": [],
        "enrollment_rate": [],
    }

    for df in results_df:
        pop = df["x0"].sum()
        offenses = df["yin"].sum()
        enrolled = df["tau"].sum()
        arrivals = df["lmd"].sum()
        departures = df["lv"].sum()

        metrics["total_population"].append(pop)
        metrics["total_offenses"].append(offenses)
        metrics["total_enrollment"].append(enrolled)
        metrics["total_arrivals"].append(arrivals)
        metrics["total_departures"].append(departures)
        metrics["offense_rate"].append(offenses / pop if pop > 0 else 0.0)
        metrics["enrollment_rate"].append(enrolled / pop if pop > 0 else 0.0)

    # Convert to numpy arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])

    return metrics
