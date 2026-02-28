import pandas as pd
import numpy as np
import torch
import joblib
import os
import glob
import hashlib
import warnings
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3: return (0, 0, 0)
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except: return (0, 0, 0)

def get_ip_id_hash(ip_str): return int(hashlib.md5(str(ip_str).encode()).hexdigest()[:15], 16)
def get_subnet_id_safe(ip_str, subnet_map): return subnet_map.get(_subnet_key(ip_str), 0)

# === 统计图标签数量 ===
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

    pkt_col = next((c for c in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets', 'Total Fwd Pkts'] if c in time_slice.columns), None)
    fwd_pkts = torch.tensor(pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values if pkt_col else np.zeros(edge_index.size(1)), dtype=torch.float)
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)

    x = torch.stack([torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum], dim=-1).float()

    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {get_ip_id_hash(ip): get_subnet_id_safe(ip, subnet_map) for ip in pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()}
        subnet_id = torch.tensor([subnet_ids_for_node.get(int(h), 0) for h in unique_nodes], dtype=torch.long)

    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'time_idx']
    edge_attr = torch.tensor(np.nan_to_num(time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values), dtype=torch.float)

    if edge_index.size(1) > 0:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=torch.tensor(labels, dtype=torch.long), n_id=n_id)
        if subnet_id is not None: data.subnet_id = subnet_id
        return data
    return None

def main():
    output_dir = "processed_data/cic_ids2017/"
    os.makedirs(output_dir, exist_ok=True)
    
    DATA_DIR = "data/2017/TrafficLabelling_"
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Please check the path.")
        return
        
    print("Loading CSVs...")
    data_frames = []
    for file_path in tqdm(csv_files, desc="Reading CSVs"):
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"Source IP": "Src IP", "Destination IP": "Dst IP", "Source Port": "Src Port", "Destination Port": "Dst Port", " Timestamp": "Timestamp"})
        if "Timestamp" in df.columns:
            data_frames.append(df)
            
    data = pd.concat(data_frames, ignore_index=True)
    del data_frames

    print("Cleaning Data...")
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"].notna() & (data["Label"] != "")]
    data = data[~data["Label"].str.lower().isin(["nan", "none"])]
    
    le = LabelEncoder()
    data["Label"] = le.fit_transform(data["Label"])
    class_names = list(le.classes_)
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    print("Processing Timestamps...")
    data["Timestamp"] = _parse_timestamp_series(data["Timestamp"])
    data.dropna(subset=["Timestamp", "Src IP", "Dst IP"], inplace=True)
    data = data.sort_values("Timestamp").reset_index(drop=True)
    data["time_idx"] = data["Timestamp"].dt.floor("20s")

    # =========================================================
    # === 关键修正：执行快照级的分层抽样切分 (Stratified Split) ===
    # =========================================================
    print("Performing Snapshot-Level Stratified Split (8:1:1)...")
    unique_times = data["time_idx"].drop_duplicates().values
    
    counts_by_time = (
        data.groupby(["time_idx", "Label"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(unique_times, fill_value=0)
    )
    
    normal_col = int(counts_by_time.sum(axis=0).idxmax())
    cols = [int(c) for c in counts_by_time.columns.tolist()]
    attack_cols = [c for c in cols if int(c) != normal_col]

    dominant_labels = []
    for t in unique_times:
        row = counts_by_time.loc[t]
        if len(attack_cols) > 0 and float(row[attack_cols].sum()) > 0.0:
            dominant_labels.append(int(row[attack_cols].idxmax()))
        else:
            dominant_labels.append(normal_col)
            
    dominant_labels = np.asarray(dominant_labels, dtype=np.int64)

    # 按照主导标签切分 time_idx
    split_seed = 42
    train_times, temp_times, _, temp_labels = train_test_split(
        unique_times,
        dominant_labels,
        test_size=0.2, # Val + Test
        stratify=dominant_labels,
        random_state=split_seed,
    )
    val_times, test_times = train_test_split(
        temp_times,
        test_size=0.5, # Val = 0.1, Test = 0.1
        stratify=temp_labels,
        random_state=split_seed,
    )

    # 抽取对应的数据框并按时间重新排序，保证序列有序性
    train_df = data[data["time_idx"].isin(train_times)].sort_values("Timestamp").copy()
    val_df = data[data["time_idx"].isin(val_times)].sort_values("Timestamp").copy()
    test_df = data[data["time_idx"].isin(test_times)].sort_values("Timestamp").copy()
    del data
    # =========================================================

    print("Normalizing...")
    feat_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c not in ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port', 'time_idx']]
    
    for df in [train_df, val_df, test_df]:
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())
            
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
    print("IDS2017 Data Generation Complete!")

if __name__ == "__main__":
    main()