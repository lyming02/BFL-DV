import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from model import CNNLSTMNet
from tqdm import tqdm
import random
import configurations
import time
import blockchain as bc
import hashlib
import json
import gc

# ========== 参数 ==========
args = configurations.args_parser()
NUM_CLIENTS = args.num_users 
EPOCHS = args.global_epochs 
LOCAL_EPOCHS = args.local_epochs if hasattr(args, 'local_epochs') else 1
BATCH_SIZE = args.train_batch_size if hasattr(args, 'train_batch_size') else 64
LR = args.lr if hasattr(args, 'lr') else 1e-3
FL_TYPE = args.FL_TYPE if hasattr(args, 'FL_TYPE') else 'FDL'
IID = args.iid if hasattr(args, 'iid') else True
ALPHA = args.alpha if hasattr(args, 'alpha') else 100

# 分簇参数
NUM_CLUSTERS = args.num_clusters  # 可通过args传递
CLIENTS_PER_CLUSTER = NUM_CLIENTS // NUM_CLUSTERS

# 恶意节点参数
MALICIOUS_CLIENTS = args.atk_node
NUM_MALICIOUS = len(MALICIOUS_CLIENTS)

if MALICIOUS_CLIENTS is not None and len(MALICIOUS_CLIENTS) > 0:
    malicious_clients = list(MALICIOUS_CLIENTS)
else:
    malicious_clients = random.sample(range(NUM_CLIENTS), NUM_MALICIOUS) if NUM_MALICIOUS > 0 else []
print(f"恶意客户端ID: {malicious_clients}")

# ========== 数据加载与预处理 ==========
df = pd.read_csv('theft_dataset.csv')
feature_cols = [str(i) for i in range(336)]
if not all(col in df.columns for col in feature_cols):
    feature_cols = list(df.columns[:336])
X = df[feature_cols].values.astype(np.float32)
y = df['label'].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ========== 联邦学习数据分割逻辑 ==========
def split_data_iid(X, y, num_clients):
    """独立同分布数据分割 - 使用StratifiedKFold确保每个客户端类别分布相似"""
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    client_data = []
    for _, idx in skf.split(X, y):
        X_local, y_local = X[idx], y[idx]
        client_data.append((X_local, y_local))
    return client_data

def split_data_non_iid(X, y, num_clients, alpha):
    """非独立同分布数据分割 - 使用Dirichlet分布控制类别分布不均匀性"""
    num_classes = len(np.unique(y))
    client_data = []
    
    # 为每个类别生成Dirichlet分布的客户端分布比例
    np.random.seed(42)
    client_distributions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # 按类别分组数据
    class_indices = {}
    for i in range(num_classes):
        class_indices[i] = np.where(y == i)[0]
    
    # 为每个客户端分配数据
    client_indices = [[] for _ in range(num_clients)]
    
    for class_id in range(num_classes):
        # 获取当前类别的所有样本索引
        class_samples = class_indices[class_id]
        np.random.shuffle(class_samples)
        
        # 根据Dirichlet分布分配样本
        proportions = client_distributions[class_id]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_samples)).astype(int)[:-1]
        proportions = np.append(proportions, len(class_samples))
        
        # 分配样本到各个客户端
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = proportions[client_id]
            client_indices[client_id].extend(class_samples[start_idx:end_idx])
            start_idx = end_idx
    
    # 创建客户端数据
    for client_id in range(num_clients):
        if len(client_indices[client_id]) > 0:
            X_local = X[client_indices[client_id]]
            y_local = y[client_indices[client_id]]
            client_data.append((X_local, y_local))
        else:
            # 如果某个客户端没有数据，随机分配一些数据
            random_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
            X_local = X[random_indices]
            y_local = y[random_indices]
            client_data.append((X_local, y_local))
    
    return client_data

# 根据IID参数选择数据分割方式
if IID:
    print("使用独立同分布(IID)数据分割")
    client_data = split_data_iid(X_train, y_train, NUM_CLIENTS)
else:
    print(f"使用非独立同分布(Non-IID)数据分割，alpha={ALPHA}")
    client_data = split_data_non_iid(X_train, y_train, NUM_CLIENTS, ALPHA)

# 打印每个客户端的数据分布信息
print("\n各客户端数据分布:")
for i, (X_local, y_local) in enumerate(client_data):
    unique, counts = np.unique(y_local, return_counts=True)
    print(f"客户端{i}: 样本数={len(y_local)}, 类别分布={dict(zip(unique, counts))}")

# ========== 初始化全局模型 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = CNNLSTMNet(input_len=336, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
start_time = time.strftime("%Y%m%d_%H%M") 
print(start_time)

# ========== 联邦学习主循环 ==========
if FL_TYPE == 'FDL':
    for epoch in tqdm(range(EPOCHS), desc="联邦学习轮次"):
        client_models = []
        print(f"\n--- 联邦第{epoch+1}轮 ---")
        for client_id, (X_local, y_local) in enumerate(tqdm(client_data, desc="客户端训练", leave=False)):
            X_local_tensor = torch.tensor(X_local).unsqueeze(1)
            # 恶意节点标签反转
            if client_id in malicious_clients and random.random() < 0.5:
                y_local = 1 - y_local
                print(f"  客户端{client_id}为恶意节点，执行标签反转攻击")
            y_local_tensor = torch.tensor(y_local)
            train_dataset = TensorDataset(X_local_tensor, y_local_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            local_model = CNNLSTMNet(input_len=336, num_classes=2).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)

            local_model.train()
            for _ in range(LOCAL_EPOCHS):
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = local_model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
            client_models.append(local_model)
        # FedAvg参数聚合
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(NUM_CLIENTS)], 0).mean(0)
        global_model.load_state_dict(global_dict)
        # 测试全局模型
        global_model.eval()
        X_test_tensor = torch.tensor(X_test).unsqueeze(1).to(device)
        y_test_tensor = torch.tensor(y_test).to(device)
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), 256):
                xb = X_test_tensor[i:i+256]
                out = global_model(xb)
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        acc = accuracy_score(y_test, all_preds)
        auc = roc_auc_score(y_test, all_probs[:, 1])
        cm = confusion_matrix(y_test, all_preds)
        
        # 计算测试集loss
        test_loss = 0.0
        num_batches = 0
        global_model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), 256):
                xb = X_test_tensor[i:i+256]
                yb = y_test_tensor[i:i+256]
                out = global_model(xb)
                loss = criterion(out, yb)
                test_loss += loss.item()
                num_batches += 1
        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"Test Acc: {acc:.4f}, Test AUC: {auc:.4f}")
        print(f"测试集Loss: {avg_test_loss:.6f}")
        print(f"混淆矩阵:\n{cm}")
        
        # 保存详细结果到csv（追加模式）
        round_result = {
            'epoch': epoch + 1,
            'acc': acc,
            'auc': auc,
            'test_loss': avg_test_loss,
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1],
            'precision': cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
            'recall': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
            'f1_score': 2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]) if (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]) > 0 else 0,
            'confusion_matrix': cm.tolist()
        }
        
        # 先尝试读取已存在的csv，再追加写入
        import os
        csv_path = f'结果/all/FDL_results{start_time}.csv'
        if os.path.exists(csv_path):
            df_results = pd.read_csv(csv_path)
            df_results = pd.concat([df_results, pd.DataFrame([round_result])], ignore_index=True)
        else:
            df_results = pd.DataFrame([round_result])
        df_results.to_csv(csv_path, index=False)
        print(f"已将本轮结果追加保存到 {csv_path}")

elif FL_TYPE == 'Cluster':
    # 分簇
    atk_model = []
    atk1_model = []
    Threshold = args.Threshold
    thresholdMultiplier = args.thresholdMultiplier
    threshold_list = []
    clusters = [list(range(i*CLIENTS_PER_CLUSTER, (i+1)*CLIENTS_PER_CLUSTER)) for i in range(NUM_CLUSTERS)]
    blockchain = bc.Blockchain()
    blockchain2 = bc.Blockchain() # 用于存储acc
    cluster_blockchains = [bc.Blockchain() for _ in range(NUM_CLUSTERS)]
    block2_index = 0
    results = []  # 用于保存每轮结果
    # 在Cluster模式下，定义每个簇的上一轮委员会成员和声誉分
    prev_committee_ids_per_cluster = [None for _ in range(NUM_CLUSTERS)]
    prev_reputations_per_cluster = [None for _ in range(NUM_CLUSTERS)]
    for epoch in tqdm(range(EPOCHS), desc="分簇顺序联邦轮次"):
        print(f"\n--- 分簇顺序联邦第{epoch+1}轮 ---")
        for cluster_id, cluster in enumerate(clusters):
            print(f"  当前簇: {cluster_id+1}/{NUM_CLUSTERS}")
            # 簇内FedAvg聚合：每个客户端本地训练后聚合
            cluster_client_models = []
            for client_id in cluster:
                X_local, y_local = client_data[client_id]
                # 恶意节点标签反转
                if client_id in malicious_clients :
                    y_local = 1 - y_local
                X_local_tensor = torch.tensor(X_local).unsqueeze(1)
                y_local_tensor = torch.tensor(y_local)
                train_dataset = TensorDataset(X_local_tensor, y_local_tensor)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                local_model = CNNLSTMNet(input_len=336, num_classes=2).to(device)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)
                local_model.train()
                if args.lr_scheduler:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
                for _ in range(LOCAL_EPOCHS):
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        out = local_model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        optimizer.step()
                cluster_client_models.append(local_model)
                if epoch == 0:
                    atk_model.append(local_model)
            # FedAvg聚合簇内模型

            cluster_dict = cluster_client_models[0].state_dict()

                # 2. 计算每个节点的声誉分
                reputation_alpha = 0.5  # 历史权重
                loss_beta = 1.0         # 损失权重

                # 1. 先收集所有节点的acc和loss
                all_accs = []
                all_losses = []
                for idx, model in enumerate(cluster_client_models):
                    accs = []
                    losses = []
                    for eval_id in committee_ids:
                        X_eval, y_eval = client_data[eval_id]
                        # # 只采样部分数据用于评估
                        # if len(X_eval) > 1000:
                        #     idx = np.random.choice(len(X_eval), 1000, replace=False)
                        #     X_eval = X_eval[idx]
                        #     y_eval = y_eval[idx]
                        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).unsqueeze(1).to(device)
                        y_eval_tensor = torch.tensor(y_eval, dtype=torch.long).to(device)
                        if eval_id < 10 and args.atk_mode['atk'] == 2  and eval_id in malicious_clients and idx in malicious_clients:
                            acc = 1.0
                            loss = 0.0
                        elif eval_id < 10 and args.atk_mode['atk'] == 2 and eval_id in malicious_clients and idx not in malicious_clients:
                            acc = 0.0
                            loss = 1.0
                        else:
                            model.eval()
                            preds = []
                            losses_eval = []
                            with torch.no_grad():
                                for i in range(0, len(X_eval_tensor), 256):
                                    xb = X_eval_tensor[i:i+256]
                                    out = model(xb)
                                    pred = out.argmax(dim=1)
                                    preds.append(pred.cpu().numpy())
                                    losses_eval.append(criterion(out, y_eval_tensor[i:i+len(xb)]).item())
                            preds = np.concatenate(preds)
                            acc = accuracy_score(y_eval, preds)
                            loss = np.mean(losses_eval)
                        accs.append(acc)
                        losses.append(loss)
                    # 对每个节点，accs和losses分别是被所有委员会成员评判的结果，取平均
                    all_accs.append(np.mean(accs))
                    all_losses.append(np.mean(losses))

                all_accs = np.array(all_accs)
                all_losses = np.array(all_losses)

                # 2. 标准化
                if all_accs.max() > all_accs.min():
                    accs_norm = (all_accs - all_accs.min()) / (all_accs.max() - all_accs.min())
                else:
                    accs_norm = np.full_like(all_accs, 0.5)
                if all_losses.max() > all_losses.min():
                    losses_norm = (all_losses - all_losses.min()) / (all_losses.max() - all_losses.min())
                else:
                    losses_norm = np.full_like(all_losses, 0.5)

                # 3. 计算声誉分
                avg_reputations = accs_norm - loss_beta * losses_norm

                # 4. 历史融合
                prev_reputations = prev_reputations_per_cluster[cluster_id]
                if prev_reputations is not None:
                    avg_reputations = avg_reputations * (1 - reputation_alpha) + prev_reputations * reputation_alpha

                # 5. 标准化到0~100
                min_rep = avg_reputations.min()
                max_rep = avg_reputations.max()
                if max_rep > min_rep:
                    avg_reputations = (avg_reputations - min_rep) / (max_rep - min_rep) * 100
                else:
                    avg_reputations = np.full_like(avg_reputations, 50.0)

                # 6. 保存
                prev_reputations_per_cluster[cluster_id] = avg_reputations

                # 3. 计算平均声誉分
                mean_reputation = avg_reputations.mean()
                print("  节点声誉分:", end=' ')
                for rep in avg_reputations:
                    print(f"{round(rep, 2)}", end=' ')
                print(f"  平均声誉分: {round(mean_reputation, 2)}")

                # 4. 选出声誉分>=平均值的节点
                selected_indices = [i for i, rep in enumerate(avg_reputations) if rep >= mean_reputation]
                print(f"  参与聚合的节点: {[cluster[i] for i in selected_indices]}")

                # 5. 聚合
                if selected_indices:
                    for key in cluster_dict.keys():
                        cluster_dict[key] = torch.stack([cluster_client_models[i].state_dict()[key].float() for i in selected_indices], 0).mean(0)
                else:
                    print("  没有节点满足声誉要求，使用所有节点聚合")
                    for key in cluster_dict.keys():
                        cluster_dict[key] = torch.stack([m.state_dict()[key].float() for m in cluster_client_models], 0).mean(0)
            # 用聚合后的簇模型更新全局模型
            global_model.load_state_dict(cluster_dict)
            # 向外输出恶意的全局模型
            if epoch == 0 and cluster_id == 0:
                atk1_model.append(cluster_dict)
            if args.atk_mode['atk'] == 3 and epoch != 0 and cluster_id == 0:
                global_model.load_state_dict(atk1_model[0])
                print("模型篡改攻击")
            # 每个簇训练后立即评估一次全局模型
            global_model.eval()
            X_test_tensor = torch.tensor(X_test).unsqueeze(1).to(device)
            y_test_tensor = torch.tensor(y_test).to(device)
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(X_test_tensor), 256):
                    xb = X_test_tensor[i:i+256]
                    out = global_model(xb)
                    probs = torch.softmax(out, dim=1)
                    preds = out.argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)
            acc = accuracy_score(y_test, all_preds)
            auc = roc_auc_score(y_test, all_probs[:, 1])
            cm = confusion_matrix(y_test, all_preds)
            print(f"  当前簇训练后Test Acc: {acc * 100:.4f}, Test AUC: {auc * 100:.4f}")
            # 存入区块链
            block_index = epoch * NUM_CLUSTERS + cluster_id
            def get_model_hash(model_dict):
                # 将参数转为字符串后计算哈希
                param_str = json.dumps({k: v.cpu().numpy().tolist() for k, v in model_dict.items()}, sort_keys=True)
                return hashlib.sha256(param_str.encode('utf-8')).hexdigest()

            model_hash = get_model_hash(cluster_dict)
            block = bc.Block(
                block_index,
                time.time(),
                {'model_hash': model_hash},  # 只存哈希
                blockchain.get_latest_block().hash,
                0,
                float(acc),
                0
            )
            blockchain.add_block(block)
            # print(f"  已将第{block_index}号区块（簇模型）存入区块链，区块hash: {block.hash}")

            # 2. def2簇间识别
            if args.atk_mode['def2'] == 1:
                if epoch > 0:
                    print(100*(acc))
                    print(100*(blockchain2.get_latest_block().acc))
                    print(100*(acc-blockchain2.get_latest_block().acc))
                    if 100*(acc-blockchain2.get_latest_block().acc)>Threshold:
                        #满足阈值，不触发防御
                        print("准确率未下降超过阈值")
                        block2 = bc.Block(block2_index, time.time(), {'model':cluster_dict}, blockchain2.get_latest_block().hash, 0, float(acc), 0)
                        block2_index = block2_index + 1
                        blockchain2.add_block(block2)
                        threshold_list.append(0)
                    else:
                        print("准确率下降过多")
                        global_model.load_state_dict(blockchain2.get_latest_block().data['model'])
                        threshold_list.append(1)
                        acc = blockchain2.get_latest_block().acc
                else:
                    # 第一轮直接存入区块链
                    block2 = bc.Block(block2_index, time.time(), {'model':cluster_dict}, blockchain2.get_latest_block().hash, 0, float(acc), 0)
                    block2_index = block2_index + 1
                    blockchain2.add_block(block2)

        #学习率衰减
        if args.lr_scheduler:
            scheduler.step()
        if args.atk_mode['def2'] == 1:
            if epoch != 0 and sum(threshold_list[-args.num_clusters:-1])==0:
                print("连续",args.num_clusters,"轮未触发防御")
                Threshold = Threshold / thresholdMultiplier
                print(f"阈值更新为: {Threshold}")   
            elif epoch != 0 and sum(threshold_list[-args.num_clusters:-1])==args.num_clusters:
                print("连续",args.num_clusters,"轮触发防御")
                Threshold = Threshold * thresholdMultiplier
                print(f"阈值更新为: {Threshold}")   
            if Threshold < -4:
                Threshold = -4
            # elif Threshold > -0.1:
            #     Threshold = -0.1
        # 每轮结束后也可打印一次混淆矩阵
        global_model.eval()
        X_test_tensor = torch.tensor(X_test).unsqueeze(1).to(device)
        y_test_tensor = torch.tensor(y_test).to(device)
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), 256):
                xb = X_test_tensor[i:i+256]
                out = global_model(xb)
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        acc = accuracy_score(y_test, all_preds)
        auc = roc_auc_score(y_test, all_probs[:, 1])
        cm = confusion_matrix(y_test, all_preds)
        
        # 计算测试集loss
        test_loss = 0.0
        num_batches = 0
        global_model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), 256):
                xb = X_test_tensor[i:i+256]
                yb = y_test_tensor[i:i+256]
                out = global_model(xb)
                loss = criterion(out, yb)
                test_loss += loss.item()
                num_batches += 1
        avg_test_loss = test_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"混淆矩阵:\n{cm}")
        print(f"测试集Loss: {avg_test_loss:.6f}")
        
        # 保存详细结果到csv（追加模式）
        round_result = {
            'epoch': epoch + 1,
            'acc': acc,
            'auc': auc,
            'test_loss': avg_test_loss,
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1],
            'precision': cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
            'recall': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
            'f1_score': 2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]) if (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]) > 0 else 0,
            'confusion_matrix': cm.tolist()
        }
        
        # 先尝试读取已存在的csv，再追加写入
        import os
        a = args.atk_mode['atk']
        csv_path = f'results/{len(args.atk_node)/args.num_users*100}%_m={args.m_num}_num_users={args.num_users}_{start_time}.csv'
        if os.path.exists (csv_path) :
            df_results = pd.read_csv(csv_path)
            df_results = pd.concat([df_results, pd.DataFrame([round_result])], ignore_index=True)
        else:
            df_results = pd.DataFrame([round_result])
        df_results.to_csv(csv_path, index=False)
        print(f"已将本轮结果追加保存到 {csv_path}")
else:

    raise ValueError(f"未知的FL_TYPE: {FL_TYPE}") 
