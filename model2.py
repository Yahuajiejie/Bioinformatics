import constructor1
import fasterfunctioning2
from transformers import AutoModel, BertTokenizer, BertConfig, AutoTokenizer
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AdamW
from tqdm import tqdm
import random
from torch.utils.data import Subset
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support,precision_score,recall_score,f1_score
import numpy as np
import copy
import os
import sys
from Bio.Seq import Seq
import Bio.SeqIO as SeqIO
import matplotlib.pyplot as plt
from collections import Counter
from torch.optim.lr_scheduler import MultiStepLR
from math import floor
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保实时写入
    def flush(self):
        for f in self.files:
            f.flush()

# 统一使用一个tokenizer
tokenizer = AutoTokenizer.from_pretrained("./DNA_bert_6", trust_remote_code=True)
model_enc = AutoModel.from_pretrained("./DNA_bert_6", trust_remote_code=True)
label_to_index={
    'donor':1,
    'acceptor':2,
    'negative':0
}
index_to_label={
    1:'donor',
    2:'acceptor',
    0:'negative'
}

class SpliceDataset(Dataset):
    def __init__(self, tokenizer, rawdata, kmer=6, flank=50):
        self.tokenizer = tokenizer
        self.kmer = kmer
        self.max_length = flank * 2 + 1
        self.flank = flank
        self.rawdata = rawdata
        
        # 预处理：将原始数据转换为 k-mer 序列
        self.samples = []
        self.balance_sample()

    def __len__(self):
        return len(self.samples)

    def k_mer_tokenize(self, seq):
        return " ".join([seq[i:i + self.kmer] for i in range(len(seq) - self.kmer + 1)])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample['seq']
        label = sample['label']
        label = label_to_index[label]

        kmered_seq = self.k_mer_tokenize(seq)
        encoded = self.tokenizer(
            kmered_seq,
            padding='max_length',
            truncation=True,
            max_length=self.max_length+2,
            return_tensors='pt',
            add_special_tokens=True 
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

    def balance_sample(self, num_per_class=None):
        """从 rawdata 中采样平衡子集"""
        by_label = {'donor': [], 'acceptor': [], 'negative': []}
        for sample in self.rawdata:
            if sample['label'] in by_label:
                by_label[sample['label']].append(sample)

        # 检查每类是否至少存在一个样本
        min_len = min(len(by_label[label]) for label in by_label)
        if min_len == 0:
            print("⚠️ Warning: Some classes have no samples. Using all available data.")
            self.samples = self.rawdata
            return

        if num_per_class is not None:
            min_len = min(min_len, num_per_class)

        self.samples = (
            random.sample(by_label['donor'], min_len) +
            random.sample(by_label['acceptor'], min_len) +
            random.sample(by_label['negative'], min_len)
        )
        random.shuffle(self.samples)

    def add_false_positives(self, fp_samples):
        """
        添加假阳性样本到负样本集合中
        fp_samples: 被模型错误预测为阳性的序列样本列表
        """
        added = 0
        for sample in fp_samples:
            # 将假阳性样本标记为负样本
            new_sample = sample.copy()
            new_sample['label'] = 'negative'
            self.rawdata.append(new_sample)
            added += 1
        
        if added > 0:
            print(f"Added {added} false positive samples to negative set")
            self.balance_sample()  # 重新平衡数据集
            
        return added

    def add_false_negatives(self, fn_samples):
        """
        添加假阴性样本到对应的阳性类别中
        fn_samples: 被模型错误预测为阴性的真实阳性序列样本列表
        """
        added = 0
        for sample in fn_samples:
            # 保持原有的标签（应该是donor或acceptor）
            if sample['label'] in ['donor', 'acceptor']:
                self.rawdata.append(sample)
                added += 1
        
        if added > 0:
            print(f"Added {added} false negative samples to positive sets")
            self.balance_sample()  # 重新平衡数据集
            
        return added


def focal_loss(logits, labels, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = (1 - pt) ** gamma * ce_loss
    return loss.mean() if reduction == 'mean' else loss

class ResidualMLP(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_size, hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.out = nn.Linear(hidden_size//2, num_labels)
        
        self.dropout = nn.Dropout(0.2)

        # 如果残差维度不匹配，用线性层投影
        self.shortcut1 = nn.Linear(hidden_size, hidden_size)
        self.shortcut2 = nn.Linear(hidden_size, hidden_size//2)

    def forward(self, x):
        # Block 1
        residual = self.shortcut1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x + residual)  # 残差连接
        x = self.dropout(x)

        # Block 2
        residual = self.shortcut2(x)
        x = self.fc3(x)
        x = F.relu(x + residual)
        x = self.dropout(x)

        return self.out(x)
class ResidualCNN(nn.Module):
    """带残差连接的CNN，用于DNA序列分类"""
    def __init__(self, input_channels, seq_len, num_classes):
        super(ResidualCNN, self).__init__()
        self.seq_len = seq_len
        # 第一层卷积 + 残差分支
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=3, padding=1)
        self.res1 = nn.Conv1d(input_channels, 128, kernel_size=1)  # 通道对齐用

        # 第二层卷积 + 残差分支
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.res2 = nn.Conv1d(128, 256, kernel_size=1)  # 通道对齐用

        # 全连接层
        self.fc1 = nn.Linear(256 * floor(floor(seq_len/2)/2), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (B, C, L)
        # Block 1
        res = self.res1(x)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.max_pool1d(x, 2)
        res = F.max_pool1d(res, 2)
        x = x + res  # 残差加和

        # Block 2
        #print(x.shape)
        res = self.res2(x)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.max_pool1d(x, 2)
        res = F.max_pool1d(res, 2)
        x = x + res

        # 展平 + 全连接
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = self.dropout(x)
        return self.fc2(x)

class DNABERTplusCNN(nn.Module):
    def __init__(self, pretrained_model, flank=50, freeze_bert=True, num_classes=3, use_cls=False):
        super(DNABERTplusCNN, self).__init__()
        self.bert = pretrained_model
        self.freeze_bert = freeze_bert
        self.num_classes = num_classes
        self.flank = flank
        self.use_cls = use_cls

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.hidden_size = self.bert.config.hidden_size
        self.seq_len = 2 * flank + 1  # 不含CLS token

        # CNN 分支处理非CLS的token表示 (不包括第0个CLS)
        self.cnn = ResidualCNN(input_channels=self.hidden_size,
                               seq_len=self.seq_len,
                               num_classes=256)

        # 中心 token（CLS 或中间位点）FC
        #self.cls_fc = nn.Sequential(
        #    nn.Linear(self.hidden_size, 256),
        #    nn.ReLU(),
        #    nn.Dropout(0.5)
        #)

        # center 4-mer 的表示（用原始 embedding 模拟）
        self.kmer_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 最终融合
        self.final_fc = nn.Linear(256 + 256, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L+1, H)

        # 获取中心向量
        cls_or_mid = hidden[:, 0, :] if self.use_cls else hidden[:, self.flank + 1, :]
        cls_out = self.cls_fc(cls_or_mid)  # (B, 256)

        # 获取除CLS外 token 表示 (B, L, H) → 不含CLS
        token_vecs = hidden[:, 1:self.seq_len + 1, :]  # (B, L=seq_len, H)
        token_vecs = token_vecs.permute(0, 2, 1)       # (B, H, L)
        cnn_out = self.cnn(token_vecs)                # (B, 256)

        # 提取中心点3-mer（前2后1）
        # 中心点在 [flank]（因为已去掉CLS，token_vecs的索引从0开始）
        up1 = hidden[:, self.flank - 2 + 1, :]  # +1 for CLS offset
        up2 = hidden[:, self.flank - 1 + 1, :]
        center = hidden[:, self.flank + 0 + 1, :]  # 中心 token
        down1 = hidden[:, self.flank + 1 + 1, :]
        center_kmer = torch.cat([up1, up2, center,down1], dim=1)  # (B, H*3)
        kmer_out = self.kmer_fc(center_kmer)              # (B, 128)

        # 融合
        fusion = torch.cat([cnn_out,kmer_out], dim=1)  # (B, 640)
        logits = self.final_fc(fusion)
        return logits
    

class DNABERTSplice(nn.Module):
    def __init__(self,
                 pretrained_model,
                 freeze_bert=True,
                 use_cls=True,
                 num_labels=3,
                 multi_task=False):
        super(DNABERTSplice, self).__init__()

        # 加载预训练 DNABERT-6 模型
        self.bert = pretrained_model

        # 是否使用 CLS 向量（否则用中心 token）
        self.use_cls = use_cls
        self.multi_task = multi_task
        self.freeze_bert = freeze_bert
        self.num_labels = num_labels  # 分类数：3 类（donor / acceptor / negative）

        # 是否冻结 BERT 参数（可微调）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 通常为 768

        # 分类头（共享 或 多头）
        if not multi_task:
            self.classifier = ResidualMLP(hidden_size, num_labels)
        else:
            self.donor_head = ResidualMLP(hidden_size, 2)  # donor / acceptor / negative
            self.acceptor_head = ResidualMLP(hidden_size, 2)  # donor / acceptor / negative
            self.classifier = nn.Linear(4,3)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.use_cls:
            pooled = outputs.last_hidden_state[:,0,:]  # shape: (batch, hidden)
        else:
            # 取中心 token（输入长度为偶数，中心位点为 seq_len // 2）
            seq_len = input_ids.shape[1]
            center_index = seq_len // 2
            pooled = outputs.last_hidden_state[:, center_index, :]  # shape: (batch, hidden)

        if self.multi_task:
            donor_logits = self.donor_head(pooled)
            acceptor_logits = self.acceptor_head(pooled)
            # 合并 logits
            logits = self.classifier(torch.cat((donor_logits, acceptor_logits), dim=-1))  # shape: (batch, 4)
        else:
            logits = self.classifier(pooled)
        return logits
        
"""单任务三分类模式
model = DNABERTSplice(pretrained_model='zhihan1996/DNABERT-6', freeze_bert=False, use_cls=False)
logits = model(input_ids, attention_mask)  # 输出 shape: (batch, 3)
"""
"""多任务联合预测模式
model = DNABERTSplice(multi_task=True)
donor_logits, acceptor_logits = model(input_ids, attention_mask)
"""
def train_one_epoch(model, dataloader, optimizer, device,gamma=None):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch['input_ids'], batch['attention_mask'])  # (B, 3)
        loss = focal_loss(logits, batch['labels'], gamma=gamma) if gamma is not None else F.cross_entropy(logits, batch['labels'], reduction='mean')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(batch['labels'].detach().cpu())

    # 汇总指标
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(dataloader)
    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, precision, recall, f1

def evaluate(model, dataloader, device, gamma=2.0):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['input_ids'], batch['attention_mask'])
            loss = focal_loss(logits, batch['labels'], gamma=gamma) if gamma is not None else F.cross_entropy(logits, batch['labels'], reduction='mean')

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(batch['labels'].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(dataloader)
    acc = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, precision, recall, f1

def reconstruct_from_kmers(decoded_str):
    """
    从 tokenizer.decode() 得到的 'AAAAGC GCTGTT ...' 形式字符串中，
    恢复原始的连续 DNA 序列（比如 'AAAAGCTGTT...'）
    """
    kmers = decoded_str.strip().split()
    if not kmers:
        return ''
    seq = kmers[0]
    for kmer in kmers[1:]:
        seq += kmer[-1]  # overlap 合并
    return seq


def second_eval(model, dataloader, device, tokenizer, flank=50):
    model.eval()
    all_preds = []
    all_labels = []
    FP_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Second Eval"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['input_ids'], batch['attention_mask'])  # shape: (B, 3)
            preds = torch.argmax(logits, dim=-1)
            labels = batch['labels']

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            for i in range(preds.shape[0]):
                pred_label = preds[i].item()
                true_label = labels[i].item()
                if true_label == 0 and pred_label != 0:  # 假阳性：negative 被预测为 donor/acceptor
                    # 使用 decode + 重建原序列
                    decoded_str = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                    raw_seq = reconstruct_from_kmers(decoded_str)
                    FP_samples.append({'seq': raw_seq, 'label': 'negative'})

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = (all_preds == all_labels).mean()

    return acc, FP_samples
def main_for_MLP(load_data=True, save_data=True,
         flank=50, train_num_samples=10000,
         validation_sample_num=1000, sample_times=100,
         epochs=30, batch_size=32, redirect=False,epoch_range=0):

    # 日志重定向
    if redirect:
        log_file = open("output.log", "w", buffering=1)
        sys.stdout = Tee(sys.stdout, log_file)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_path = './best_dnabert_splice.pth'

    # 初始化模型
    model = DNABERTSplice(pretrained_model=model_enc, freeze_bert=False, use_cls=True,multi_task=True)
    best_f1 = 0.0
    model.to(device)

    # 加载已有模型
    if os.path.exists(save_path) and load_data:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resuming from checkpoint. Previous best F1: {best_f1:.4f}")
    print("Model loaded.")

    # 数据准备
    gtf_path = './coding_only.gtf'
    fasta_path = './GRCh38.primary_assembly.genome.fa'

    print("Generating training samples...")
    sample_train = constructor1.generally_create_train_dataset(
        gtf_path=gtf_path, fasta_path=fasta_path,
        flank=flank, num_samples=train_num_samples,
        neg_pos_per_transcript=2)

    print("Creating SpliceDataset...")
    train_valid_dataset = SpliceDataset(tokenizer, sample_train, flank=flank)

    # 加载第三阶段验证对象
    genome, splice_coords_set_donor, splice_coords_set_acceptor = constructor1.load_sampling_object(
        gtf_path, fasta_path, flank=flank)

    optimizer = AdamW(model.parameters(), lr=2e-4)
    scheduler = MultiStepLR(optimizer, milestones=[(i+1)*50 for i in range(epochs//50)], gamma=0.1)

    # 训练循环
    for epoch in range(1, epochs + 1):
        # 动态切分训练/验证集
        total_len = len(train_valid_dataset)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_ds, val_ds = random_split(train_valid_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print('='*50)

        # 第一阶段：正常训练验证
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        print(f"[Train] Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f}")
        print(f"[Val  ] Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")

        # 第二阶段：随机验证集（找 FP）
        val_samples = constructor1.create_validation_dataset(gtf_path, fasta_path, flank=flank,
                                                             num_samples=validation_sample_num)
        val_dataset2 = SpliceDataset(tokenizer, val_samples, flank=flank)
        val_loader2 = DataLoader(val_dataset2, batch_size=batch_size)
        acc2,FP = second_eval(model, val_loader2, device,tokenizer,flank=flank)
        print(f"[Random Eval] Acc={acc2:.4f} ")
        # 获取假阴性（由 second_eval 返回 FP） → 增强训练集
        train_valid_dataset.add_false_positives(FP)

        if epoch >= epoch_range:
        # 第三阶段：滑动窗口验证（真实场景）
            donor_results, FP2, FN2, TP, TN, FPn, FNn = fasterfunctioning2.ultra_sliding_eval_by_splice_anchor(
                model, tokenizer, genome,
                splice_coords_set_acceptor=splice_coords_set_acceptor,
                splice_coords_set_donor=splice_coords_set_donor,
                region_length=1000, flank=flank, device=device,
                sample_time=sample_times)

            train_valid_dataset.add_false_positives(FP2)
            train_valid_dataset.add_false_negatives(FN2)

            accuracy = (TP + TN) / (TP + TN + FPn + FNn)
            precision = TP / (TP + FPn) if (TP + FPn) > 0 else 0
            recall = TP / (TP + FNn) if (TP + FNn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            print(f"[Sliding Eval] Acc={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1_score:.4f}")

        # 保存最优模型
            if f1_score > best_f1 and save_data:
                best_f1 = f1_score
                torch.save({'model_state_dict': model.state_dict(), 'best_f1': best_f1}, save_path)
                print(f"✅ Epoch {epoch}: New best F1={f1_score:.4f}, model saved to {save_path}")

        # 打印示例预测
            for r in donor_results[:3]:
                print(f"== {r['chrom']} ({r['strand']}) ==")
                print(f"Predicted splice: {r['predictions']}")
                print(f"True splice:      {r['true_splice_coord']}")
def main_for_CNN(load_data=True, save_data=True,
         flank=50, train_num_samples=10000,
         validation_sample_num=1000, sample_times=100,
         epochs=30, batch_size=32, redirect=False,epoch_range=100):

    # 日志重定向
    if redirect:
        log_file = open("output.log", "w", buffering=1)
        sys.stdout = Tee(sys.stdout, log_file)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_path = './best_dnabert_cnn.pth'

    # 初始化模型
    model = DNABERTplusCNN(pretrained_model=model_enc, freeze_bert=False, num_classes=3, flank=flank)
    best_f1 = 0.0
    model.to(device)

    # 加载已有模型
    if os.path.exists(save_path) and load_data:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resuming from checkpoint. Previous best F1: {best_f1:.4f}")
    print("Model loaded.")

    # 数据准备
    gtf_path = './coding_only.gtf'
    fasta_path = './GRCh38.primary_assembly.genome.fa'

    print("Generating training samples...")
    sample_train = constructor1.generally_create_train_dataset(
        gtf_path=gtf_path, fasta_path=fasta_path,
        flank=flank, num_samples=train_num_samples,
        neg_pos_per_transcript=2)

    print("Creating SpliceDataset...")
    train_valid_dataset = SpliceDataset(tokenizer, sample_train, flank=flank)

    # 加载第三阶段验证对象
    genome, splice_coords_set_donor, splice_coords_set_acceptor = constructor1.load_sampling_object(
        gtf_path, fasta_path, flank=flank)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = MultiStepLR(optimizer, milestones=[(i+1)*50 for i in range(epochs//50)], gamma=0.1)

    # 训练循环
    for epoch in range(1, epochs + 1):
        # 动态切分训练/验证集
        total_len = len(train_valid_dataset)
        train_size = int(0.8 * total_len)
        val_size = total_len - train_size
        train_ds, val_ds = random_split(train_valid_dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")        
        print('='*50)
        # 第一阶段：正常训练验证
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        print(f"[Train] Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f}")
        print(f"[Val  ] Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
        # 第二阶段：随机验证集（找 FP） 
        val_samples = constructor1.create_validation_dataset(gtf_path, fasta_path, flank=flank,
                                                             num_samples=validation_sample_num)
        val_dataset2 = SpliceDataset(tokenizer, val_samples, flank=flank)
        val_loader2 = DataLoader(val_dataset2, batch_size=batch_size)           
        acc2,FP = second_eval(model, val_loader2, device,tokenizer,flank=flank)
        print(f"[Random Eval] Acc={acc2:.4f} ")
        # 获取假阴性（由 second_eval 返回 FP） → 增强训练集
        train_valid_dataset.add_false_positives(FP)  
        if epoch >= epoch_range:
            # 第三阶段：滑动窗口验证（真实场景）
            donor_results, FP2, FN2, TP, TN, FPn, FNn = fasterfunctioning2.ultra_sliding_eval_by_splice_anchor(
                model, tokenizer, genome,
                splice_coords_set_acceptor=splice_coords_set_acceptor,
                splice_coords_set_donor=splice_coords_set_donor,
                region_length=1000, flank=flank, device=device,
                sample_time=sample_times)

            train_valid_dataset.add_false_positives(FP2)
            train_valid_dataset.add_false_negatives(FN2)

            accuracy = (TP + TN) / (TP + TN + FPn + FNn)
            precision = TP / (TP + FPn) if (TP + FPn) > 0 else 0
            recall = TP / (TP + FNn) if (TP + FNn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            print(f"[Sliding Eval] Acc={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1_score:.4f}")

            # 保存最优模型
            if f1_score > best_f1 and save_data:
                best_f1 = f1_score
                torch.save({'model_state_dict': model.state_dict(), 'best_f1': best_f1}, save_path)
                print(f"✅ Epoch {epoch}: New best F1={f1_score:.4f}, model saved to {save_path}")

            # 打印示例预测
            for r in donor_results[:3]:
                print(f"== {r['chrom']} ({r['strand']}) ==")
                print(f"Predicted splice: {r['predictions']}")
                print(f"True splice:      {r['true_splice_coord']}")
            

if __name__ == "__main__":
    main_for_CNN(True,True,train_num_samples=5000,validation_sample_num=100,flank=75,sample_times=5,epochs=100,redirect=True,epoch_range=0)


"""

RuntimeError: grad can be implicitly created only for scalar outputs
是在调用 .backward() 时抛出的，意思是：
你试图对一个非标量（不是单个值）的张量求导，而 .backward() 默认只支持对标量 loss进行自动反向传播。


常见原因：自定义损失函数没有 mean()
你很可能在使用 focal loss 或 cross_entropy(..., reduction='none') 的版本，比如：
你很可能在使用 focal loss 或 cross_entropy(..., reduction='none') 的版本，比如：

python
复制
编辑
ce_loss = F.cross_entropy(logits, labels, reduction='none')
pt = torch.exp(-ce_loss)
loss = (1 - pt) ** gamma * ce_loss
这时候 loss 是一个 向量，shape 为 (batch,)。

所以你需要明确指定如何将多个 loss 聚合为一个标量，例如：

python
复制
编辑
return loss.mean()  # 或 loss.sum()

"""