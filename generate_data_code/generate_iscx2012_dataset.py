import pandas as pd
import numpy as np
import torch
import joblib
import os
import re
import hashlib
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm

def _iscxids2012_sort_key(path):
    m = re.search(r"Jun(\d{1,2})", os.path.basename(path))
    return int(m.group(1)) if m else os.path.basename(path)

def _basic_time_and_label(df):
    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).lower() == "normal" else 1)

    for col in ["startDateTime", "stopDateTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["startDateTime", "stopDateTime"])
    df["Duration of time"] = (df["stopDateTime"] - df["startDateTime"]).dt.total_seconds() / 60.0
    df["time_window"] = df["startDateTime"].dt.floor("min")
    df = df.drop(columns=["stopDateTime", "startDateTime", "generated"], errors="ignore")
    return df

def get_ip_id_hash(ip_str):
    return int(hashlib.md5(str(ip_str).encode()).hexdigest()[:15], 16)

# === 统计图中的边标签数量 ===
def _edge_label_counts_from_graphs(graph_seq, num_classes=2):
    counts = np.zeros(num_classes, dtype=np.int64)
    for g in graph_seq:
        if g is None: continue
        labels = g.edge_labels.detach().cpu().numpy().astype(np.int64)
        if labels.size == 0: continue
        counts += np.bincount(labels, minlength=num_classes).astype(np.int64)
    return counts

def print_graph_label_stats(graph_seq, split_name, class_names):
    counts = _edge_label_counts_from_graphs(graph_seq, len(class_names))
    stats = [f"{class_names[i]}({i}): {counts[i]}" for i in range(len(class_names))]
    print(f"[{split_name} Graphs] {len(graph_seq)} graphs, Edge Label Counts -> " + ", ".join(stats))

def create_graph_data_inductive_2012(time_slice):
    time_slice = time_slice.copy()
    time_slice["source"] = time_slice["source"].astype(str).str.strip()
    time_slice["destination"] = time_slice["destination"].astype(str).str.strip()

    src_ips = time_slice["source"].to_numpy()
    dst_ips = time_slice["destination"].to_numpy()

    all_nodes = np.concatenate([src_ips, dst_ips], axis=0)
    unique_nodes, inverse_indices = np.unique(all_nodes, return_inverse=True)

    n_nodes = int(len(unique_nodes))
    src_local = inverse_indices[: len(src_ips)]
    dst_local = inverse_indices[len(src_ips) :]

    edge_index = torch.tensor(np.stack([src_local, dst_local], axis=0), dtype=torch.long)
    n_id = torch.tensor([get_ip_id_hash(ip) for ip in unique_nodes], dtype=torch.long)

    if edge_index.size(1) <= 0 or n_nodes <= 0: return None

    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)

    src_port = pd.to_numeric(time_slice.get("sourcePort", 0), errors="coerce").fillna(0).to_numpy()
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    priv_port_count.scatter_add_(0, torch.tensor(src_local, dtype=torch.long), torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    pkt_vals = pd.to_numeric(time_slice.get("totalSourcePackets", 0), errors="coerce").fillna(0).to_numpy()
    fwd_pkts = torch.tensor(pkt_vals, dtype=torch.float)
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    node_pkt_sum.scatter_add_(0, torch.tensor(src_local, dtype=torch.long), fwd_pkts)

    x = torch.stack([torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum], dim=1).float()
    labels = torch.tensor(time_slice["Label"].values.astype(np.int64), dtype=torch.long)

    drop_cols = ["appName", "source", "destination", "Label", "time_window", "sourcePort", "destinationPort"]
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).values.astype(np.float32)
    edge_attr = torch.tensor(np.nan_to_num(edge_attr_vals), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=labels, n_id=n_id)
    # 附加 time_window 用于后续 Fallback 反向切分表格数据
    data.time_window = time_slice["time_window"].iloc[0] 
    return data

def main():
    output_dir = "processed_data/iscx_ids2012/"
    os.makedirs(output_dir, exist_ok=True)
    
    base_path = "data/ISCXIDS2012"
    csv_paths = sorted([os.path.join(base_path, p) for p in os.listdir(base_path) if p.lower().endswith(".csv")], key=_iscxids2012_sort_key)
    
    if len(csv_paths) < 6:
        raise FileNotFoundError("Need at least 6 CSV files in ISCXIDS2012 directory.")
        
    train_paths = csv_paths[:4]
    val_path = csv_paths[4]
    test_path = csv_paths[5]
    
    print("Loading CSVs...")
    train_raw = pd.concat([_basic_time_and_label(pd.read_csv(p, low_memory=False)) for p in train_paths], ignore_index=True)
    val_raw = _basic_time_and_label(pd.read_csv(val_path, low_memory=False))
    test_raw = _basic_time_and_label(pd.read_csv(test_path, low_memory=False))

    print("Encoding and Scaling...")
    train_df, val_df, test_df = train_raw.copy(), val_raw.copy(), test_raw.copy()

    drop_cols = [c for c in train_df.columns[train_df.notna().mean() <= 0.3] if c not in ["Label", "time_window"]]
    train_df.drop(columns=drop_cols, errors="ignore", inplace=True)
    val_df.drop(columns=drop_cols, errors="ignore", inplace=True)
    test_df.drop(columns=drop_cols, errors="ignore", inplace=True)

    if "sourceTCPFlagsDescription" in train_df.columns and "destinationTCPFlagsDescription" in train_df.columns:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        tcp_cols = ["sourceTCPFlagsDescription", "destinationTCPFlagsDescription"]
        enc.fit(train_df[tcp_cols].fillna(""))
        for df in (train_df, val_df, test_df):
            df_tcp = pd.DataFrame(enc.transform(df[tcp_cols].fillna("")), columns=enc.get_feature_names_out(tcp_cols), index=df.index)
            df.drop(columns=tcp_cols, inplace=True, errors="ignore")
            df[df_tcp.columns] = df_tcp

    for col in ["protocolName", "sourcePayloadAsUTF", "destinationPayloadAsUTF", "direction"]:
        if col in train_df.columns:
            mapping = {v: i + 1 for i, v in enumerate(sorted(set(train_df[col].astype(str).fillna("").tolist())))}
            for df in (train_df, val_df, test_df):
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna("").map(mapping).fillna(0).astype(np.int64)

    if "sourcePayloadAsBase64" in train_df.columns:
        for df in (train_df, val_df, test_df):
            df["sourcePayloadLength"] = df["sourcePayloadAsBase64"].apply(lambda x: len(str(x)))
            df["destinationPayloadLength"] = df["destinationPayloadAsBase64"].apply(lambda x: len(str(x)))
            df.drop(columns=["sourcePayloadAsBase64", "destinationPayloadAsBase64"], inplace=True, errors="ignore")

    feat_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in {"Label", "sourcePort", "destinationPort", "totalSourcePackets", "totalDestinationPackets", "time_window"}]

    for df in (train_df, val_df, test_df):
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in feat_cols:
        if train_df[col].max() > 100:
            for df in (train_df, val_df, test_df): df[col] = np.log1p(df[col].abs())

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print("Building Graphs...")
    train_seq = [create_graph_data_inductive_2012(g) for _, g in tqdm(train_df.groupby("time_window", sort=True))]
    val_seq = [create_graph_data_inductive_2012(g) for _, g in tqdm(val_df.groupby("time_window", sort=True))]
    test_seq = [create_graph_data_inductive_2012(g) for _, g in tqdm(test_df.groupby("time_window", sort=True))]

    train_seq = [g for g in train_seq if g]
    val_seq = [g for g in val_seq if g]
    test_seq = [g for g in test_seq if g]

    class_names = ["Normal", "Attack"]
    print("\n--- Initial Graph Statistics ---")
    print_graph_label_stats(train_seq, "Train", class_names)
    print_graph_label_stats(val_seq, "Val", class_names)
    print_graph_label_stats(test_seq, "Test", class_names)

    # ==========================================
    # === 关键步骤：执行 Val Fallback 切分逻辑 ===
    # ==========================================
    min_val_attack_edges = 50
    val_counts = _edge_label_counts_from_graphs(val_seq, num_classes=2)
    
    if int(val_counts[1]) < min_val_attack_edges:
        print(f"\n⚠️ Val split has too few Attack edges ({val_counts[1]} < {min_val_attack_edges}); fallback to split from Train+Val sequence.")
        
        combined = list(train_seq) + list(val_seq)
        n = len(combined)
        if n >= 2:
            start = max(1, int(n * 0.9))
            chosen_start = start
            best_candidate = None
            best_candidate_attack = -1
            
            while chosen_start > 1:
                candidate = combined[chosen_start:]
                c = _edge_label_counts_from_graphs(candidate, num_classes=2)
                cand_attack = int(c[1])
                
                if cand_attack > best_candidate_attack:
                    best_candidate_attack = cand_attack
                    best_candidate = chosen_start
                if cand_attack >= min_val_attack_edges:
                    break
                    
                step = max(1, int(n * 0.05))
                chosen_start = max(1, chosen_start - step)
                
            if best_candidate is not None:
                chosen_start = best_candidate
                
            train_seq = combined[:chosen_start]
            val_seq = combined[chosen_start:]
            
            print("--- Fallback Graph Statistics ---")
            print_graph_label_stats(train_seq, "Train", class_names)
            print_graph_label_stats(val_seq, "Val", class_names)

            # === 同步修正 DataFrame (为了保证 .npz 数据的一致性) ===
            # 取新 val_seq 第一张图携带的 time_window 作为新的分割线
            if len(val_seq) > 0:
                new_split_time = val_seq[0].time_window
                print(f"-> Resplitting DataFrames at time: {new_split_time}")
                
                combined_df = pd.concat([train_df, val_df])
                train_df = combined_df[combined_df["time_window"] < new_split_time].copy()
                val_df = combined_df[combined_df["time_window"] >= new_split_time].copy()

    # ==========================================

    print("\nSaving flattened data (synchronized with graphs)...")
    np.savez(os.path.join(output_dir, "flattened_data.npz"),
             X_train=train_df[feat_cols].values, y_train=train_df['Label'].values,
             X_val=val_df[feat_cols].values, y_val=val_df['Label'].values,
             X_test=test_df[feat_cols].values, y_test=test_df['Label'].values,
             feature_names=feat_cols)

    torch.save(train_seq, os.path.join(output_dir, "train_graphs.pt"))
    torch.save(val_seq, os.path.join(output_dir, "val_graphs.pt"))
    torch.save(test_seq, os.path.join(output_dir, "test_graphs.pt"))

    print("ISCX2012 Data Generation Complete!")

if __name__ == "__main__":
    main()