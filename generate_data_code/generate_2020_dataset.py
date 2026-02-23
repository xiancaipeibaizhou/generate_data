import pandas as pd
import numpy as np
import torch
import joblib
import os
import hashlib
import warnings
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3: return (0, 0, 0)
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except: return (0, 0, 0)

def get_ip_id_hash(ip_str):
    return int(hashlib.md5(str(ip_str).encode()).hexdigest()[:15], 16)

def get_subnet_id_safe(ip_str, subnet_map):
    return subnet_map.get(_subnet_key(ip_str), 0)

# 打印图数据中的标签统计
def print_graph_label_stats(graph_seq, split_name, class_names):
    counts = np.zeros(len(class_names), dtype=np.int64)
    for g in graph_seq:
        if g is not None and g.edge_labels is not None:
            labels = g.edge_labels.detach().cpu().numpy().astype(np.int64)
            if len(labels) > 0:
                counts += np.bincount(labels, minlength=len(class_names))
    
    stats = [f"{class_names[i]}({i}): {counts[i]}" for i in range(len(class_names)) if counts[i] > 0]
    print(f"[{split_name} Graphs] Edge Label Counts -> " + ", ".join(stats))

def _parse_timestamp_series(ts):
    ts = ts.astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(ts, errors="coerce", dayfirst=True)

def create_graph_data_inductive(time_slice, subnet_map):
    time_slice = time_slice.copy()
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    src_ids = time_slice['Src IP'].apply(get_ip_id_hash).values.astype(np.int64)
    dst_ids = time_slice['Dst IP'].apply(get_ip_id_hash).values.astype(np.int64)
    labels = time_slice['Label'].values.astype(int)

    all_nodes = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        out_degrees.scatter_add_(0, edge_index[0], ones)
        in_degrees.scatter_add_(0, edge_index[1], ones)

    src_port_col = 'Src Port' if 'Src Port' in time_slice.columns else 'Source Port'
    src_port = pd.to_numeric(time_slice.get(src_port_col, 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    pkt_col = next((cand for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets'] if cand in time_slice.columns), None)
    if pkt_col is None:
        fwd_pkts = torch.zeros(edge_index.size(1), dtype=torch.float)
    else:
        fwd_pkts = torch.tensor(pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values, dtype=torch.float)
    
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)

    # a2_4.py 特有：is_hub 节点特征 (总维度 5)
    is_hub = ((in_degrees + out_degrees) > 50).to(torch.float)
    x = torch.stack([torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum, is_hub], dim=-1).float()

    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {get_ip_id_hash(ip): get_subnet_id_safe(ip, subnet_map) for ip in pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()}
        subnet_id = torch.tensor([subnet_ids_for_node.get(int(h), 0) for h in unique_nodes], dtype=torch.long)

    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'time_idx', 'block_id']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=torch.tensor(labels, dtype=torch.long), n_id=n_id)
        if subnet_id is not None: data.subnet_id = subnet_id
        return data
    return None

def main():
    output_dir = "processed_data/darknet2020_block/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Darknet2020 Data...")
    data = pd.read_csv("data/CIC-Darknet2020/Darknet.csv")
    data.drop(columns=['Label.1'], inplace=True, errors='ignore')
    data.dropna(subset=['Label', 'Timestamp'], inplace=True)
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"] != ""]
    data = data[~data["Label"].str.lower().isin(["nan", "none"])]
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    
    print("Processing Time...")
    data['Timestamp'] = _parse_timestamp_series(data['Timestamp'])
    data.dropna(subset=['Timestamp'], inplace=True)
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('min')

    # ================================================================
    # 严格对齐 a2_4.py 的 Adaptive Block 切分逻辑
    # ================================================================
    print("Performing Adaptive Block-wise Split...")
    unique_times = data["time_idx"].drop_duplicates().values
    total_len = len(unique_times)
    num_blocks = 10
    min_block_times = 30
    require_all_classes = False

    counts_by_time_all = (
        data.groupby(["time_idx", "Label"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(unique_times, fill_value=0)
    )
    counts_mat_all = counts_by_time_all.to_numpy(dtype=np.int64, copy=False)

    idx_splits = []
    start = 0
    current_counts = np.zeros(counts_mat_all.shape[1], dtype=np.int64)
    for i in range(total_len):
        current_counts += counts_mat_all[i]
        cur_len = i - start + 1
        if len(idx_splits) >= (num_blocks - 1): continue
        has_all = bool(np.all(current_counts > 0))
        if cur_len >= min_block_times and ((not require_all_classes) or has_all):
            idx_splits.append(np.arange(start, i + 1, dtype=np.int64))
            start = i + 1
            current_counts = np.zeros(counts_mat_all.shape[1], dtype=np.int64)
    if start < total_len:
        idx_splits.append(np.arange(start, total_len, dtype=np.int64))
    if len(idx_splits) == 0:
        idx_splits = [np.arange(0, total_len, dtype=np.int64)]

    # 寻找包含所有标签类别的 Target Block (即 a2_4.py 运行过程中有效率最高的 block)
    target_block_idx = None
    for b, idxs in enumerate(idx_splits):
        block_times = unique_times[idxs]
        block_df = data[data["time_idx"].isin(block_times)]
        if len(block_df['Label'].unique()) == len(class_names):
            target_block_idx = b
            print(f"Found Block {b} containing all {len(class_names)} classes.")
    
    if target_block_idx is None:
        target_block_idx = 8
        print(f"Warning: No block contains all classes. Defaulting to block {target_block_idx}")
    
    # 提取目标 Block 的数据
    block_times = unique_times[idx_splits[target_block_idx]]
    block_df = data[data["time_idx"].isin(block_times)].copy()
    block_len = len(block_times)

    # ================================================================
    # 严格对齐 a2_4.py 的 Stratified-Time Inner Split (贪心分配算法)
    # ================================================================
    print("Performing Stratified-Time Inner Split (Greedy Assignment)...")
    block_train_ratio = 0.8
    block_val_ratio = 0.1
    min_val_classes = len(class_names)
    min_test_classes = len(class_names)
    min_val_per_class = 20
    min_test_per_class = 20

    base_train_idx = int(block_len * block_train_ratio)
    base_val_idx = int(block_len * (block_train_ratio + block_val_ratio))
    base_train_idx = max(1, min(block_len - 2, base_train_idx))
    base_val_idx = max(base_train_idx + 1, min(block_len - 1, base_val_idx))

    counts_by_time = (
        block_df.groupby(["time_idx", "Label"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(block_times, fill_value=0)
    )
    counts_mat = counts_by_time.to_numpy(dtype=np.int64, copy=False)
    prefix = np.cumsum(counts_mat, axis=0, dtype=np.int64)
    prefix0 = np.vstack([np.zeros((1, prefix.shape[1]), dtype=np.int64), prefix])
    total_counts = prefix0[-1]

    n_train = int(base_train_idx)
    n_val = int(max(1, base_val_idx - base_train_idx))
    n_test = int(max(1, block_len - base_val_idx))
    n_total = int(n_train + n_val + n_test)
    if n_total > block_len:
        overflow = n_total - block_len
        n_train = max(1, n_train - overflow)

    totals = total_counts.astype(np.float64, copy=False)
    rarity = 1.0 / (totals + 1.0)
    time_scores = counts_mat @ rarity
    order = np.argsort(-time_scores, kind="mergesort")

    caps = {"train": int(n_train), "val": int(n_val), "test": int(n_test)}
    chosen = {"train": [], "val": [], "test": []}
    set_counts = {
        "train": np.zeros(counts_mat.shape[1], dtype=np.int64),
        "val": np.zeros(counts_mat.shape[1], dtype=np.int64),
        "test": np.zeros(counts_mat.shape[1], dtype=np.int64),
    }
    assigned = set()

    def _eligible(counts, min_cls, min_per):
        ok = counts >= int(min_per) if int(min_per) > 0 else counts > 0
        return int(np.sum(ok)) >= int(min_cls)

    def _gain(name, vec, min_cls, min_per):
        counts = set_counts[name]
        if int(min_per) > 0:
            need = np.maximum(0, int(min_per) - counts)
            add = np.minimum(vec, need)
            gain = float(np.sum(rarity * (add > 0)))
        else:
            missing = (counts == 0) & (vec > 0)
            gain = float(np.sum(rarity[missing]))
        fill = len(chosen[name]) / float(max(1, caps[name]))
        return gain - 0.05 * fill

    for name, min_cls, min_per in (("val", min_val_classes, min_val_per_class), ("test", min_test_classes, min_test_per_class)):
        if caps[name] <= 0: continue
        while (not _eligible(set_counts[name], min_cls, min_per)) and (len(chosen[name]) < caps[name]):
            best_idx, best_gain = None, -1e9
            for idx in order.tolist():
                if idx in assigned: continue
                vec = counts_mat[idx]
                if vec.sum() <= 0: continue
                g = _gain(name, vec, min_cls, min_per)
                if g > best_gain:
                    best_gain, best_idx = g, idx
            if best_idx is None or best_gain <= 0: break
            chosen[name].append(best_idx)
            assigned.add(best_idx)
            set_counts[name] = set_counts[name] + counts_mat[best_idx]

    for idx in order.tolist():
        if idx in assigned: continue
        candidates = [k for k in ("train", "val", "test") if len(chosen[k]) < caps[k]]
        if not candidates: break
        vec = counts_mat[idx]
        if vec.sum() <= 0:
            best = min(candidates, key=lambda k: len(chosen[k]) / float(max(1, caps[k])))
        else:
            best = max(candidates, key=lambda k: _gain(k, vec, 
                min_val_classes if k == "val" else (min_test_classes if k == "test" else 0),
                min_val_per_class if k == "val" else (min_test_per_class if k == "test" else 0)))
        chosen[best].append(idx)
        assigned.add(idx)
        set_counts[best] = set_counts[best] + vec

    remaining = [i for i in range(block_len) if i not in assigned]
    for name in ("train", "val", "test"):
        need = caps[name] - len(chosen[name])
        if need > 0 and remaining:
            chosen[name].extend(remaining[:need])
            for j in remaining[:need]: set_counts[name] = set_counts[name] + counts_mat[j]
            remaining = remaining[need:]

    train_times = block_times[np.array(sorted(chosen["train"]), dtype=np.int64)]
    val_times = block_times[np.array(sorted(chosen["val"]), dtype=np.int64)]
    test_times = block_times[np.array(sorted(chosen["test"]), dtype=np.int64)]

    train_df = block_df[block_df["time_idx"].isin(train_times)].copy()
    val_df = block_df[block_df["time_idx"].isin(val_times)].copy()
    test_df = block_df[block_df["time_idx"].isin(test_times)].copy()
    del data

    # ================================================================
    # 归一化 (与原代码一致)
    # ================================================================
    print("Normalizing...")
    feat_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port', 'time_idx', 'block_id']]
    
    for df in [train_df, val_df, test_df]:
        for col in feat_cols:
            train_col = pd.to_numeric(train_df[col], errors="coerce")
            finite_mask = np.isfinite(train_col.to_numpy(dtype=np.float64, copy=False))
            if finite_mask.any():
                finite_max = float(np.max(train_col.to_numpy(dtype=np.float64, copy=False)[finite_mask]))
                finite_min = float(np.min(train_col.to_numpy(dtype=np.float64, copy=False)[finite_mask]))
            else:
                finite_max, finite_min = 0.0, 0.0
            df[col] = df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)
        df[feat_cols] = df[feat_cols].fillna(0)
    
    for col in feat_cols:
        if train_df[col].max() > 100:
            for df in [train_df, val_df, test_df]: df[col] = np.log1p(df[col].abs())

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    print("Saving flattened data...")
    np.savez(os.path.join(output_dir, "flattened_data.npz"),
             X_train=train_df[feat_cols].values, y_train=train_df['Label'].values,
             X_val=val_df[feat_cols].values, y_val=val_df['Label'].values,
             X_test=test_df[feat_cols].values, y_test=test_df['Label'].values,
             feature_names=feat_cols)

    print("Building Subnet Map...")
    subnet_to_idx = {'<UNK>': 0}
    for ip in pd.concat([train_df['Src IP'], train_df['Dst IP']]).astype(str).str.strip().unique():
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)

    print("Building Graphs...")
    train_seq = [create_graph_data_inductive(group, subnet_to_idx) for _, group in tqdm(train_df.groupby('time_idx', sort=True), desc="Train")]
    val_seq = [create_graph_data_inductive(group, subnet_to_idx) for _, group in tqdm(val_df.groupby('time_idx', sort=True), desc="Val")]
    test_seq = [create_graph_data_inductive(group, subnet_to_idx) for _, group in tqdm(test_df.groupby('time_idx', sort=True), desc="Test")]

    train_seq = [g for g in train_seq if g]
    val_seq = [g for g in val_seq if g]
    test_seq = [g for g in test_seq if g]

    print("\n--- Graph Statistics ---")
    print_graph_label_stats(train_seq, "Train", class_names)
    print_graph_label_stats(val_seq, "Val", class_names)
    print_graph_label_stats(test_seq, "Test", class_names)
    print("------------------------\n")

    torch.save(train_seq, os.path.join(output_dir, "train_graphs.pt"))
    torch.save(val_seq, os.path.join(output_dir, "val_graphs.pt"))
    torch.save(test_seq, os.path.join(output_dir, "test_graphs.pt"))
    print("Darknet2020 Block Data Generation Complete!")

if __name__ == "__main__":
    main()