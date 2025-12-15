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
    """
    Compute trajectory statistics using arrest count differences between periods.

    For each period p (starting from 1):
    - x0: population at start of period (from snapshot p-1), grouped by state at p-1
    - x: population at end of period (from snapshot p), grouped by state at p
    - y: offenses = j_p - j_{p-1} (difference in felony_arrest)
    - y_out: offenses aggregated by state at p-1 (outflow from state)
    - y_in: offenses aggregated by state at p (inflow to state)
    """
    snaps = len(self.t_dfi)
    results = []
    results_df = []
    results_flow = []
    results_retention = []
    error_accumulated = 0.0
    # start from p=1 since we need p-1 for comparison
    pbar = _tqdm(range(1, snaps), desc="Summarizing trajectory")

    for p in pbar:
        # df_prev: snapshot at p-1 (start of period)
        # df_curr: snapshot at p (end of period)
        df_prev = (
            self.t_dfi[p - 1]
            .copy()
            .dropna(subset=["state", "state_lst"])
            .assign(
                state=lambda df: df["state"].apply(lambda x: x.value),
                state_lst=lambda df: df["state_lst"].apply(lambda x: x.value),
            )
        )
        df_curr = (
            self.t_dfi[p]
            .copy()
            .dropna(subset=["state", "state_lst"])
            .assign(
                state=lambda df: df["state"].apply(lambda x: x.value),
                state_lst=lambda df: df["state_lst"].apply(lambda x: x.value),
            )
        )

        lst_cols_avail = [c for c in state_lst_columns if c in df_prev.columns]
        cur_cols_avail = [c for c in state_columns if c in df_curr.columns]
        grp_keys_lst = lst_cols_avail
        grp_keys_cur = cur_cols_avail

        _boundary = p + p_freeze
        _boundary_prev = (p - 1) + p_freeze

        # x0: present at start of period (using p-1 snapshot), grouped by state at p-1
        _present_start_mask = (df_prev["ep_arrival"] <= _boundary_prev) & (
            df_prev["ep_leaving"] > _boundary_prev
        )
        df_presence_begin = df_prev.loc[_present_start_mask]
        x0 = df_presence_begin.groupby(grp_keys_cur)["index"].count()

        # x: present at end of period (using p snapshot), grouped by state at p
        _present_end_mask = (df_curr["ep_arrival"] <= _boundary) & (
            df_curr["ep_leaving"] > _boundary
        )
        x = df_curr.loc[_present_end_mask].groupby(grp_keys_cur)["index"].count()

        # -----------------------------------------------------------------
        # Compute y using arrest count difference: j_p - j_{p-1}
        # -----------------------------------------------------------------
        # Get individuals present in both periods
        idx_common = df_presence_begin.index.intersection(df_curr.index)

        # Create merged df with state at p-1, state at p, and felony_arrest at both
        df_merged = pd.DataFrame(
            {
                "index": idx_common,
                "state_prev": df_prev.loc[idx_common, "state"],
                "state_curr": df_curr.loc[idx_common, "state"],
                "j_prev": df_prev.loc[idx_common, "felony_arrest"],
                "j_curr": df_curr.loc[idx_common, "felony_arrest"],
            }
        ).assign(offenses=lambda df: df["j_curr"] - df["j_prev"])

        # y_out: offenses aggregated by state at p-1 (outflow from state)
        yout = (
            df_merged[df_merged["offenses"] > 0]
            .groupby("state_prev")["offenses"]
            .sum()
            .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
        )

        # y_in: offenses aggregated by state at p (inflow to state)
        yin = (
            df_merged[df_merged["offenses"] > 0]
            .groupby("state_curr")["offenses"]
            .sum()
            .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
        )

        # arrivals and departures
        # arrivals: people arriving at end of period (ep_arrival == _boundary)
        _arrive_now_mask = df_curr["ep_arrival"] == _boundary
        # departures: use df_curr for ep_leaving (set when leaving event processed)
        _leave_now_mask = df_curr["ep_leaving"] == _boundary
        lmd = df_curr.loc[_arrive_now_mask].groupby(grp_keys_cur)["index"].count()
        # lv: people present at start who leave at boundary
        _leave_indices = df_presence_begin.index.intersection(
            df_curr[_leave_now_mask].index
        )
        lv = (
            df_presence_begin.loc[_leave_indices].groupby(grp_keys_cur)["index"].count()
        )

        # inc: incarcerated during the period (type_left == 3 means INCARCERATION)
        _incarcerated_indices = df_presence_begin.index.intersection(
            df_curr[_leave_now_mask & (df_curr["type_left"] == 3)].index
        )
        inc = (
            df_presence_begin.loc[_incarcerated_indices]
            .groupby(grp_keys_cur)["index"]
            .count()
        )

        # nr: number of returns during the period (difference in return_times)
        if "return_times" in df_prev.columns and "return_times" in df_curr.columns:
            df_returns = pd.DataFrame(
                {
                    "index": idx_common,
                    "state_curr": df_curr.loc[idx_common, "state"],
                    "rt_prev": df_prev.loc[idx_common, "return_times"],
                    "rt_curr": df_curr.loc[idx_common, "return_times"],
                }
            ).assign(returns=lambda df: df["rt_curr"] - df["rt_prev"])
            nr = (
                df_returns[df_returns["returns"] > 0]
                .groupby("state_curr")["returns"]
                .sum()
                .rename_axis(grp_keys_cur[0] if grp_keys_cur else "state")
            )
        else:
            nr = pd.Series(dtype=float)

        # tau: treatment in period
        tau = df_presence_begin.groupby(grp_keys_cur)["bool_treat"].sum()

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
                    "inc": inc,
                    "nr": nr,
                }
            )
            .fillna(0)
            .rename_axis(index=grp_keys_cur)
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
        pbar.set_postfix(error_max_average=f"{error_accumulated / p:.2f}")

        # Store y flow information (state_prev -> state_curr for offenders)
        y_flow = (
            df_merged[df_merged["offenses"] > 0]
            .groupby(["state_prev", "state_curr"])["offenses"]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "state_prev": "state_lst",
                    "state_curr": "state",
                    "offenses": "count",
                }
            )
        )

        rr = {
            "y": (
                y_flow.set_index(["state_lst", "state"])["count"].to_dict()
                if len(y_flow) > 0
                else {}
            ),
            **df_traj.to_dict(),
        }
        results.append(rr)
        results_df.append(df_traj)

        # compute exact flows from previous state(s) to current state(s)
        try:
            # x_flow: individuals present at start who did NOT have offenses
            x_flow_df = (
                df_merged[df_merged["offenses"] == 0]
                .groupby(["state_prev", "state_curr"])["index"]
                .count()
                .rename("count")
                .reset_index()
                .rename(columns={"state_prev": "state_lst", "state_curr": "state"})
            )
            x_flow_df["source"] = "x"
            _xtot = x_flow_df.groupby(["state_lst"])["count"].transform("sum")
            x_flow_df["ratio"] = np.where(_xtot > 0, x_flow_df["count"] / _xtot, 0.0)

            # y_flow: individuals present at start who had offenses
            y_flow_df = (
                df_merged[df_merged["offenses"] > 0]
                .groupby(["state_prev", "state_curr"])["offenses"]
                .sum()
                .rename("count")
                .reset_index()
                .rename(columns={"state_prev": "state_lst", "state_curr": "state"})
            )
            y_flow_df["source"] = "y"
            _ytot = y_flow_df.groupby(["state_lst"])["count"].transform("sum")
            y_flow_df["ratio"] = np.where(_ytot > 0, y_flow_df["count"] / _ytot, 0.0)

            # arrivals as separate source with previous-state marked 'NEW'
            arrival_flow_df = (
                df_curr.loc[_arrive_now_mask]
                .groupby(grp_keys_cur)["index"]
                .count()
                .rename("count")
                .reset_index()
            )
            arrival_flow_df["state_lst"] = "NEW"
            arrival_flow_df["source"] = "new"
            _ntot_new = arrival_flow_df.groupby(["state_lst"])["count"].transform("sum")
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
            flow_df = pd.DataFrame(
                columns=["state_lst", "state", "count", "source", "ratio", "ratio_all"]
            )

        # compute ratios: within-source and across all sources per origin
        if len(flow_df) > 0:
            _src_tot = flow_df.groupby(["state_lst", "source"], group_keys=False)[
                "count"
            ].transform("sum")
            flow_df["ratio"] = np.where(_src_tot > 0, flow_df["count"] / _src_tot, 0.0)
            _orig_tot = flow_df.groupby(["state_lst"], group_keys=False)[
                "count"
            ].transform("sum")
            flow_df["ratio_all"] = np.where(
                _orig_tot > 0, flow_df["count"] / _orig_tot, 0.0
            )

        # sort by source, previous-state columns, then current-state columns
        sort_cols = ["source", "state_lst", "state"]
        sort_cols = [c for c in sort_cols if c in flow_df.columns]
        if len(sort_cols) > 0:
            flow_df = flow_df.sort_values(by=sort_cols).reset_index(drop=True)

        flow_df["snap"] = p
        results_flow.append(flow_df)

        # compute retention per origin based on starts (including arrivals) and leaves
        _start_mask_union = _present_start_mask | (
            df_prev["ep_arrival"] == _boundary_prev
        )
        _starts = df_prev.loc[_start_mask_union].groupby(grp_keys_cur)["index"].count()
        # _leaves: people who started and left at boundary (use df_curr for leave mask)
        _start_indices = df_prev[_start_mask_union].index
        _leave_start_indices = _start_indices.intersection(
            df_curr[_leave_now_mask].index
        )
        _leaves = (
            df_prev.loc[_leave_start_indices].groupby(grp_keys_cur)["index"].count()
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
        # rename index column to match expected structure
        if grp_keys_cur:
            _ret_df = _ret_df.rename(columns={grp_keys_cur[0]: "state_lst"})
        retention_df = _ret_df.assign(
            state_lst_key=lambda df: df["state_lst"].apply(
                self.state_defs.state_key_range.get
            ),
        )

        results_retention.append(retention_df)

    # compute moving-average counts first, then ratios over last N episodes
    if len(results_flow) > 0:
        _all_flow = pd.concat(results_flow, ignore_index=True)
        needed_cols = ["state_lst", "state", "source", "snap", "count"]
        if all(c in _all_flow.columns for c in needed_cols):
            _all_flow = _all_flow.sort_values(
                by=["state_lst", "state", "source", "snap"]
            )
            # rolling average counts per origin+source+destination
            _all_flow["count_ma"] = (
                _all_flow.groupby(["state_lst", "state", "source"], group_keys=False)[
                    "count"
                ]
                .rolling(window=windowsize, min_periods=10)
                .mean()
                .reset_index(level=[0, 1, 2], drop=True)
            )
            # per-source normalization across destinations
            _all_flow["origin_source_total_ma"] = _all_flow.groupby(
                ["state_lst", "source", "snap"], group_keys=False
            )["count_ma"].transform("sum")
            _all_flow["ratio_ma"] = np.where(
                _all_flow["origin_source_total_ma"] > 0,
                _all_flow["count_ma"] / _all_flow["origin_source_total_ma"],
                0.0,
            )
            # across all sources normalization
            _all_flow["origin_total_ma"] = _all_flow.groupby(
                ["state_lst", "snap"], group_keys=False
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
                c in _all_ret.columns for c in ["state_lst", "snap", "stay", "left"]
            ):
                _all_ret = _all_ret.sort_values(by=["state_lst", "snap"])
                # rolling average counts then compute ratio
                _all_ret["stay_ma"] = (
                    _all_ret.groupby(["state_lst"], group_keys=False)["stay"]
                    .rolling(window=windowsize, min_periods=10)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                _all_ret["left_ma"] = (
                    _all_ret.groupby(["state_lst"], group_keys=False)["left"]
                    .rolling(window=windowsize, min_periods=10)
                    .mean()
                    .reset_index(level=0, drop=True)
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
    except Exception:
        pass

    # sum of last 20 results
    from functools import reduce

    if len(results_df) >= windowsize:
        mean_df = reduce(
            lambda x, y: x.add(y, fill_value=0), results_df[-windowsize:]
        ).astype(float)
        mean_df = mean_df / windowsize
    elif len(results_df) > 0:
        mean_df = reduce(lambda x, y: x.add(y, fill_value=0), results_df).astype(float)
        mean_df = mean_df / len(results_df)
    else:
        mean_df = pd.DataFrame()

    # build transition matrices Px and Py (state_lst_key -> state_key)
    if len(results_flow) > 0:
        _all_flow = pd.concat(results_flow, ignore_index=True)
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
                dfw = dfw.dropna(subset=["state_key", "state_lst_key"])
                if len(dfw) == 0:
                    return
                grouped = dfw.groupby(["state_lst_key", "state_key"], as_index=False)[
                    "ratio_ma"
                ].mean()
                row_sum = grouped.groupby("state_lst_key")["ratio_ma"].sum().to_dict()
                for _, r in grouped.iterrows():
                    i = int(r["state_lst_key"])
                    j = int(r["state_key"])
                    s = float(row_sum.get(i, 0.0))
                    p = (float(r["ratio_ma"]) / s) if s > 0 else 0.0
                    if 0 <= i < num_states and 0 <= j < num_states:
                        M[i, j] = p

            _fill_matrix(flow_last[flow_last["source"] == "x"], Px)
            _fill_matrix(flow_last[flow_last["source"] == "y"], Py)

            self.Px = Px
            self.Py = Py
            self.Px_vec = Px.flatten()
            self.Py_vec = Py.flatten()

    # save x, x0, lmd, lv, yout, yin, tau, inc, nr from mean_df
    try:
        _md = mean_df.reset_index()
        # use "state" column for mapping (current state at end of period)
        if "state" in _md.columns:
            _md["state_key"] = _md["state"].map(self.state_defs.state_key_range)
        num_states = max(self.state_defs.state_key_range.values()) + 1

        def _to_vec(column_name: str) -> np.ndarray:
            vec = np.zeros(num_states, dtype=float)
            if column_name in _md.columns and "state_key" in _md.columns:
                for _, r in _md.dropna(subset=["state_key"]).iterrows():
                    k = int(r["state_key"]) if not pd.isna(r["state_key"]) else -1
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
        self.inc_vec = _to_vec("inc")
        self.nr_vec = _to_vec("nr")
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
            state_key=lambda df: (
                df["state"].map(self.state_defs.state_key_range)
                if "state" in df.columns
                else None
            ),
        ),
        results,
        results_df,
        results_flow,
        results_retention,
    )


def summarize_trajectory_bak(
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
        x0, x, lmd, lv, yin, yout, tau, inc, nr, tau_rel, xcal, error

    Example:
        state_lst     x0    x  lmd   lv  yin  yout  tau  inc  nr  tau_rel  xcal  error
    (25.0, 4.0, 1.0)  1.0  1.0  0.0  0.0  1.0   1.0  1.0  0.0 0.0 0.999001   1.0    0.0
    ...

    Returns dict with vectors (one value per episode):
        - total_population: total individuals present (sum of x0)
        - total_offenses: total recidivism events (sum of yin)
        - total_enrollment: total treated individuals (sum of tau)
        - total_arrivals: new arrivals (sum of lmd)
        - total_departures: left system (sum of lv)
        - total_incarcerated: incarcerated during period (sum of inc)
        - total_returns: number of returns during period (sum of nr)
        - offense_rate: offenses / population
        - enrollment_rate: enrolled / population
        - incarceration_rate: incarcerated / population
        - return_rate: returns / population
    """
    metrics = {
        "total_population": [],
        "total_offenses": [],
        "total_enrollment": [],
        "total_arrivals": [],
        "total_departures": [],
        "total_incarcerated": [],
        "total_returns": [],
        "offense_rate": [],
        "enrollment_rate": [],
        "incarceration_rate": [],
        "return_rate": [],
    }

    for df in results_df:
        pop = df["x0"].sum()
        offenses = df["yin"].sum()
        enrolled = df["tau"].sum()
        arrivals = df["lmd"].sum()
        departures = df["lv"].sum()
        incarcerated = df["inc"].sum() if "inc" in df.columns else 0.0
        returns = df["nr"].sum() if "nr" in df.columns else 0.0

        metrics["total_population"].append(pop)
        metrics["total_offenses"].append(offenses)
        metrics["total_enrollment"].append(enrolled)
        metrics["total_arrivals"].append(arrivals)
        metrics["total_departures"].append(departures)
        metrics["total_incarcerated"].append(incarcerated)
        metrics["total_returns"].append(returns)
        metrics["offense_rate"].append(offenses / pop if pop > 0 else 0.0)
        metrics["enrollment_rate"].append(enrolled / pop if pop > 0 else 0.0)
        metrics["incarceration_rate"].append(incarcerated / pop if pop > 0 else 0.0)
        metrics["return_rate"].append(returns / pop if pop > 0 else 0.0)

    # Convert to numpy arrays
    for k in metrics:
        metrics[k] = np.array(metrics[k])

    # Save treatment decisions per episode (flattened for HDF5 storage)
    # Collect per-episode arrays
    index_list = []
    tau_list = []
    tau_rel_list = []
    lengths = []

    for df in results_df:
        # Index is state tuple - convert to 2D array (n_states x n_dims)
        idx_arr = np.array(
            [list(i) if hasattr(i, "__iter__") else [i] for i in df.index]
        )
        tau = df["tau"].values if "tau" in df.columns else np.zeros(len(df))
        tau_rel = df["tau_rel"].values if "tau_rel" in df.columns else np.zeros(len(df))
        index_list.append(idx_arr)
        tau_list.append(tau)
        tau_rel_list.append(tau_rel)
        lengths.append(len(df))

    # Flatten into arrays for storage
    metrics["treatment_index_flat"] = (
        np.vstack(index_list) if index_list else np.array([])
    )  # shape: (total_states, n_dims)
    metrics["treatment_tau_flat"] = (
        np.concatenate(tau_list) if tau_list else np.array([])
    )
    metrics["treatment_tau_rel_flat"] = (
        np.concatenate(tau_rel_list) if tau_rel_list else np.array([])
    )
    metrics["treatment_lengths"] = np.array(
        lengths
    )  # to reconstruct per-episode arrays

    return metrics


def recover_treatment_decision_as_df(metrics, rep=None):
    """
    Recover treatment decisions as a DataFrame with MultiIndex.

    Args:
        metrics: dict from evaluation_metrics() or all_metrics[policy_name]
        rep: repetition index (required if metrics has multiple reps, i.e. from all_metrics)

    Returns:
        pd.DataFrame with columns ['tau', 'tau_rel'] and MultiIndex (state, p)
        If multiple reps and rep=None, includes 'rep' in the index: (rep, state, p)
    """
    import pandas as pd

    index_flat = metrics["treatment_index_flat"]
    tau_flat = metrics["treatment_tau_flat"]
    tau_rel_flat = metrics["treatment_tau_rel_flat"]
    lengths = metrics["treatment_lengths"]

    # Check if this is multi-rep data (list of arrays) or single-rep (single array)
    is_multi_rep = isinstance(lengths, list)

    if is_multi_rep:
        if rep is not None:
            # Extract single repetition
            index_flat = index_flat[rep]
            tau_flat = tau_flat[rep]
            tau_rel_flat = tau_rel_flat[rep]
            lengths = lengths[rep]
        else:
            # Combine all reps with rep index
            dfs = []
            for r in range(len(lengths)):
                idx = index_flat[r]
                tau = tau_flat[r]
                tau_rel = tau_rel_flat[r]
                lens = lengths[r]

                states = [tuple(row) for row in idx]
                episodes = np.repeat(np.arange(len(lens)), lens)
                reps = np.full(len(states), r)

                multi_idx = pd.MultiIndex.from_arrays(
                    [reps, states, episodes], names=["rep", "state", "p"]
                )
                df = pd.DataFrame({"tau": tau, "tau_rel": tau_rel}, index=multi_idx)
                dfs.append(df)
            return pd.concat(dfs)

    # Single rep case
    states = [tuple(row) for row in index_flat]
    episodes = np.repeat(np.arange(len(lengths)), lengths)

    multi_idx = pd.MultiIndex.from_arrays([states, episodes], names=["state", "p"])
    df = pd.DataFrame({"tau": tau_flat, "tau_rel": tau_rel_flat}, index=multi_idx)

    return df


def compute_equilibrium_treatment_stats(all_metrics, metric_key="tau", last_n=10):
    """
    Compute equilibrium average of a metric over the last N episodes across all reps.
    Only considers states where bool_in_probation == 1 (last index level), then drops it.

    Args:
        all_metrics: dict from read_metrics_from_h5 containing treatment fields:
            - treatment_index_flat: list of arrays (one per rep) with state tuples
            - treatment_tau_flat: list of arrays (one per rep)
            - treatment_tau_rel_flat: list of arrays (one per rep)
            - treatment_lengths: list of arrays (one per rep) with episode lengths
        metric_key: "tau" or "tau_rel" (default "tau")
        last_n: number of episodes from the end to average over (default 10)

    Returns:
        (df_mean, df_std): two DataFrames with:
            - rows = age (second dim of state)
            - columns = number of offenses (first dim of state)
            - df_mean: normalized so all cells sum to 1
            - df_std: each cell scaled by its mean value (coefficient of variation)
    """
    index_flat_list = all_metrics["treatment_index_flat"]
    tau_flat_list = all_metrics["treatment_tau_flat"]
    tau_rel_flat_list = all_metrics["treatment_tau_rel_flat"]
    lengths_list = all_metrics["treatment_lengths"]

    n_reps = len(lengths_list)

    # Collect per-state data across all reps
    # state_data[state] = [val1, val2, ...] where each entry is one rep's average
    state_data = defaultdict(list)

    for rep in range(n_reps):
        index_flat = index_flat_list[rep]
        tau_flat = tau_flat_list[rep]
        tau_rel_flat = tau_rel_flat_list[rep]
        lengths = lengths_list[rep]

        # Select metric
        if metric_key == "tau":
            metric_flat = tau_flat
        else:
            metric_flat = tau_rel_flat

        n_episodes = len(lengths)
        if n_episodes < last_n:
            ep_start = 0
        else:
            ep_start = n_episodes - last_n

        # Compute cumulative indices to slice into flat arrays
        cum_lengths = np.cumsum(lengths)
        start_indices = np.concatenate([[0], cum_lengths[:-1]])

        # Collect data for last_n episodes, grouped by state (excluding bool_in_probation)
        state_metric = defaultdict(list)

        for ep in range(ep_start, n_episodes):
            start_idx = int(start_indices[ep])
            end_idx = int(cum_lengths[ep])

            for i in range(start_idx, end_idx):
                state_tuple = tuple(index_flat[i])
                # Filter: only in probation (last index == 1)
                if state_tuple[-1] != 1:
                    continue
                # Drop last index level (bool_in_probation)
                state_key = state_tuple[:-1]
                state_metric[state_key].append(metric_flat[i])

        # Average over episodes for this rep, store per state
        for state_key in state_metric:
            state_data[state_key].append(np.mean(state_metric[state_key]))

    # Build result: rows = age, columns = offenses
    # state_key = (offenses, age)
    offenses_set = set()
    age_set = set()
    for state_key in state_data:
        offenses_set.add(state_key[0])
        age_set.add(state_key[1])

    offenses_sorted = sorted(offenses_set)
    age_sorted = sorted(age_set)

    # Create mean and std matrices
    mean_data = {off: {} for off in offenses_sorted}
    std_data = {off: {} for off in offenses_sorted}

    for state_key, values in state_data.items():
        offenses, age = state_key
        arr = np.array(values)
        mean_data[offenses][age] = np.mean(arr)
        std_data[offenses][age] = np.std(arr)

    # Build DataFrames: rows = age, columns = offenses
    df_mean_raw = pd.DataFrame(
        mean_data, index=age_sorted, columns=offenses_sorted
    ).fillna(0)
    df_std_raw = pd.DataFrame(
        std_data, index=age_sorted, columns=offenses_sorted
    ).fillna(0)

    # Normalize mean_df so all cells sum to 1
    total_sum = df_mean_raw.values.sum()
    df_mean = df_mean_raw / total_sum if total_sum > 0 else df_mean_raw

    # Scale std_df by mean values (coefficient of variation)
    df_std = df_std_raw / df_mean_raw.replace(0, np.nan)

    df_mean.index.name = "age"
    df_mean.columns.name = "offenses"
    df_std.index.name = "age"
    df_std.columns.name = "offenses"

    return df_mean, df_std, df_mean_raw, df_std_raw
