import constructor1
import functioning
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
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import copy
import os
import sys
from Bio.Seq import Seq
import Bio.SeqIO as SeqIO
import matplotlib.pyplot as plt
from collections import Counter

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
    def __init__(self,tokenizer,samples,kmer=6,flank=50):
        self.tokenizer = tokenizer
        self.kmer=kmer
        self.max_length = flank*2+1
        self.samples = samples
        self.flank = flank
    def __len__(self):
        return len(self.samples)
    def k_mer_tokenize(self, seq):
        return " ".join([seq[i:i + self.kmer] for i in range(len(seq) - self.kmer + 1)])
    def __getitem__(self,idx):
        sample = self.samples[idx]
        seq = sample['seq']
        label = sample['label']
        label = label_to_index[label]
        kmered_seq = self.k_mer_tokenize(seq)
        encoded = self.tokenizer(
            kmered_seq,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }
#这是 PyTorch 的一个重要机制：DataLoader 可以直接处理 __getitem__ 返回字典的 Dataset，
#并自动将它们组合成 batch 字典，前提是所有 value 的张量 shape 可对齐。

    def add_sample(self,new_samples):
        self.samples.extend(new_samples)
        
class DNABERT_Splice(nn.Module):
    def __init__(self,encoder,num_labels=3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.Linear = nn.Linear(self.encoder.config.hidden_size,self.encoder.config.hidden_size*2)
        self.ReLU = nn.ReLU()
        self.classifier = nn.Linear(self.encoder.config.hidden_size*2,num_labels) 
    def forward(self, input_ids, attention_mask,labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    # 改为取中间 token
        mid_idx = input_ids.shape[1] // 2
        middle_output = outputs.last_hidden_state[:, mid_idx, :]  # (B, H)
        middle_output = self.Linear(middle_output)
        middle_output = self.ReLU(middle_output)
        logits = self.classifier(middle_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 注意：labels shape 应为 (B, L)，每个位置一个类别
            loss = loss_fct(logits.view(-1, logits.shape[-1]),labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0 
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch) ##解压字典
        loss = output['loss'] 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss
def first_eval(model, dataloader, device,flank=50):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch) ##解压字典
        
            logits = output['logits']  # shape: (B, 3)
            preds = torch.argmax(logits, dim=-1)  # shape: (B )
            labels = batch['labels']  

            all_preds.append(preds)
            all_labels.append(labels)
     # 拼接所有 batch 的预测与标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).sum().item() / len(all_labels)
    return acc
def second_eval(model, dataloader, device,flank=50):
    model.eval()
    all_preds = []
    all_labels = []
    FP = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch) ##解压字典
        
            logits = output['logits']  # shape: (B, 3)
            preds = torch.argmax(logits, dim=-1)  # shape: (B )
            labels = batch['labels']  


            for i in range(preds.shape[0]):
                if preds[i] != 0 and labels[i] == 0:
                    FP.append(
                    {
                    'seq': str(tokenizer.decode(batch['input_ids'][i])), #, skip_special_tokens=True
                    'label': 'negative',#index_to_label[labels[i].item()], # 从tensor变为整数
                    'pos': flank,  # 表示剪切点在序列中间
                    })
            all_preds.append(preds)
            all_labels.append(labels)
     # 拼接所有 batch 的预测与标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).sum().item() / len(all_labels)
    return acc,FP
def sliding_eval_by_splice_anchor(model, tokenizer, genome, splice_coords_set_donor,splice_coords_set_acceptor,device, mode='donor',region_length=1000, flank=50, kmer=6, sample_time=1):
    model.eval()
    results = []
    window_size = flank * 2 + 1
    current_times = 0 #采样次数
    FP_sample,FN_sample = [],[]
    TPS,FPS,TNS,FNS =0,0,0,0
    splice_coords_set = splice_coords_set_donor if mode == 'donor' else splice_coords_set_acceptor
    with torch.no_grad():
        for chrom, center, strand in random.sample(list(splice_coords_set), k=20):  # 可选随机下采样数量
            if current_times >= sample_time:
                break
            chrom_seq = genome[chrom].seq
            region_length_pre = random.randint(0,region_length // 2)
            region_length_post = region_length - region_length_pre
            seq_start = center - region_length_pre - flank
            seq_end = center + region_length_post + flank
            
            if seq_start < 0 or seq_end >= len(chrom_seq):
                continue
            seq_length = seq_end-seq_start # 2025/6/8/13/31 seq_length 不用+1
            seq = str(chrom_seq[seq_start:seq_end])
            if strand == '-':
                seq = str(Seq(seq).reverse_complement())

            absolute_coords = []
            tokenized_chunks = []
            for i in range(flank, seq_length - flank):
                window_seq = seq[i - flank: i + flank + 1]
                kmer_seq = " ".join([window_seq[j:j + kmer] for j in range(len(window_seq) - kmer + 1)])
                tokenized_chunks.append(kmer_seq)
                if strand == '+':
                    coord = seq_start + i  # 正链：基因组上等于起点向右偏移 i
                else:
                    coord = seq_end - i - 1  # 负链：反向偏移，注意是 end - i - 1
                absolute_coords.append(coord)

            # 批量编码
            encodings = tokenizer(tokenized_chunks, padding='max_length', truncation=True, max_length=window_size, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            pred_splice_coords = []
            for i, p in enumerate(preds):
                if p in (1, 2):  # 预测为 donor 或 acceptor
                    pred_splice_coords.append({
                'coord': absolute_coords[i],
                'label': 'donor' if p == 1 else 'acceptor'
                })

            # 获取输入序列的正确答案
            # 提取真实剪接位点（可选）
            true_splice_in_region = []
            for coord in absolute_coords:
                if (chrom, coord, strand) in splice_coords_set_donor:
                    true_splice_in_region.append({'coord':coord,'label':'donor'})
                    # 提取预测为剪接位点的区域位点
                elif (chrom, coord, strand) in splice_coords_set_acceptor:
                    true_splice_in_region.append({'coord':coord,'label':'acceptor'})
            # 存储每个位点的预测结果
            # 识别假阳性
            for pred in pred_splice_coords:
                # 如果预测的剪接位点不在真实剪接位点集合中，则为假阳性
                if not any(true_coord['coord'] == pred['coord'] for true_coord in true_splice_in_region):
                    FP_sample.append({
                        'seq': str(chrom_seq[pred['coord']-flank:pred['coord']+flank+1]),  # 完整序列
                        'label': 'negative', # 从tensor变为整数
                        'pos': flank,
                    })
                    FPS +=1
                else:
                    TPS +=1
            for truth in true_splice_in_region:
                if not any(truth['coord'] == pred['coord'] for pred in pred_splice_coords):
                    FN_sample.append({
                        'seq': str(chrom_seq[pred['coord']-flank:pred['coord']+flank+1]),  # 完整序列
                        'label': 'negative', # 从tensor变为整数
                        'pos': flank,
                    })
                    FNS+=1
            TNS = region_length - FPS - TPS - FNS
            result_entry = {
                "chrom": chrom,
                "strand": strand,
                "predictions": pred_splice_coords,#central_preds,
                "true_splice_coord": true_splice_in_region
            }
            results.append(result_entry)
            current_times += 1

    return results,FP_sample,FN_sample,TPS,FPS,TNS,FNS
def main(load_data = True,save_data=True,flank=50,train_num_samples= 10000,validation_sample_num = 1000, sample_times=100, epochs = 30,
    batch_size = 32,redirect=False):
    # 参数设置
    if redirect == True:
        file_path = "output.log"
        if os.path.exists(file_path):
            os.remove(file_path)
        log_file = open("output.log", "w", buffering=1)
        sys.stdout = Tee(sys.stdout, log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    save_path = './mymodel2.pth'

    # ----- construct the model ----- #
    model = DNABERT_Splice(encoder=model_enc)
    best_f1 =0.0#best_acc = 0.0
    best_model_state = None

    # 读取现有模型参数
    if os.path.exists(save_path) and load_data:
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from checkpoint. Previous best_acc: {best_acc}")
    # 选择设备
    model.to(device)
    print("Model successfully loaded and moved to device")
    # ----- load the data and create the sample() ----- #
    # 加载训练数据
    gtf_path = './coding_only.gtf'
    fasta_path = './GRCh38.primary_assembly.genome.fa'

    flank=50
    train_neg_pos_per_transcript=2
    sample_train=constructor1.generally_create_train_dataset(gtf_path=gtf_path,fasta_path=fasta_path 
                                                            ,flank=flank,num_samples=train_num_samples,
                                                            neg_pos_per_transcript=train_neg_pos_per_transcript)                                               
    print("Creating datasets...")
    train_valid_dataset = SpliceDataset(tokenizer,sample_train,flank=flank)

    # 训练集验证集划分
    print(f"Train dataset size: {int(train_num_samples*0.8)}")
    print(f"Eval dataset size: {train_num_samples-int(train_num_samples*0.8)}")
    
    
    # 训练器和数据
    
    optimizer = AdamW(model.parameters(), lr=2e-5)


    # 提前对第三轮验证进行采样
    genome, splice_coords_set_donor, splice_coords_set_acceptor = constructor1.load_sampling_object(gtf_path, fasta_path, flank=50)
    # 注意 第三轮验证的集合需要增加donor和acceptor的信息

    # 训练和评估
    for epoch in range(epochs):
        train_sample_num , valid_sample_num = int(train_valid_dataset.__len__()*0.8 ), train_valid_dataset.__len__() -int(train_valid_dataset.__len__()*0.8 )
        train_dataset,valid_dataset=random_split(train_valid_dataset, [train_sample_num,valid_sample_num])


        # 第一次验证：固定验证集+训练集
        train_loader,eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True),DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        

        # 第二次验证：随机验证数据集
        validation_sample = constructor1.create_validation_dataset(gtf_path=gtf_path,fasta_path=fasta_path,flank=flank,num_samples=validation_sample_num)
        eval_dataset = SpliceDataset(tokenizer, validation_sample,flank=flank)
        second_eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*50)
        
        # 训练
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Train_loss{train_loss:.4f}")
        # 评估
        print("\nEvaluating...")
        # CrossValidation
        acc1 = first_eval(model,eval_loader,device)

        #if acc1 > best_acc and save_data:

        print(f"Cross Validation Acc: {acc1:.4f}")
        # RandomValidation
        acc2,FP = second_eval(model, second_eval_loader, device)
        train_valid_dataset.add_sample(FP)#train_dataset.add_sample(FP) # 自定义dataset的子set不会继承自定义函数
        print(f"Random Eval Acc: {acc2:.4f}")
        # Scanning Validation

        donor_results,FP2,FN2,TPs,TNs,FPs,FNs = functioning.sliding_eval_by_splice_anchor_optimized(model, tokenizer, genome, 
                                                        splice_coords_set_acceptor=splice_coords_set_acceptor
                                                        ,splice_coords_set_donor=splice_coords_set_donor,
                                                        region_length=1000, flank=50, device=device,sample_time=sample_times)
        train_valid_dataset.add_sample(FP2)
        train_valid_dataset.add_sample(FN2)
        
        accuracy = (TPs + TNs) / (TPs + TNs + FPs + FNs)
        precision = TPs / (TPs + FPs) if TPs + FPs > 0 else 0
        recall = TPs / (TPs + FNs) if TPs + FNs > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        if f1_score > best_f1 and save_data:
            best_f1 = f1_score
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch}: New best F1 = {f1_score}, model saved.")
            if best_model_state is not None:
                torch.save({
                    'model_state_dict': best_model_state,
                    'best_f1': best_f1}
                    ,save_path)
                print(f"Best model saved to {save_path}")
            else:
                print("No model was saved. Check training process.")

        # 可视化预测 or 存储结果
        for r in donor_results[:3]:
            print(f"== {r['chrom']} ({r['strand']}) ==")
            print(f"Pred splice coord: {r['predictions']}")
            print(f"True splice coord: {r['true_splice_coord']}")
if __name__ == "__main__":
    main(True,False,train_num_samples=10000,validation_sample_num=100,sample_times=1,epochs=10)