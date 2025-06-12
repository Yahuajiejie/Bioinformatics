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
from Bio.Seq import Seq
import Bio.SeqIO as SeqIO
import matplotlib.pyplot as plt
from collections import Counter

#保存路径重定向
import sys

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
    def __init__(self,tokenizer,samples,max_sample_per_class=10000,kmer=6,flank=50):
        self.tokenizer = tokenizer
        self.max_samples_per_class = max_sample_per_class
        self.kmer=kmer
        self.max_length = flank*2+1
        self.rawdata = samples
        self.flank = flank
        self.samples = self._balance_samples(self.rawdata)
    def _balance_samples(self, samples):
        """平衡各类别样本数量"""
        if self.max_samples_per_class is None:
            return samples
            
        # 按类别分组
        samples_by_class = {'donor': [], 'acceptor': [], 'negative': []}
        for sample in samples:
            samples_by_class[sample['label']].append(sample)
        
        # 限制每个类别的最大样本数
        balanced_samples = []
        for label, class_samples in samples_by_class.items():
            if len(class_samples) > self.max_samples_per_class:
                import random
                class_samples = random.sample(class_samples, self.max_samples_per_class)
            balanced_samples.extend(class_samples)
        
        return balanced_samples    
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
        self.rawdata.extend(new_samples)
        self.samples = self._balance_samples(self.rawdata)

class DNABERT_Splice(nn.Module):
    def __init__(self,encoder,num_labels=3,dropout_rate=0.6,focal_alpha = torch.tensor([0.1, 0.45, 0.45]), focal_gamma = 1.5):
        super().__init__()
        self.encoder = encoder
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size *2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder.config.hidden_size *2, num_labels)
        )
    def focal_loss(self, inputs, targets, alpha=None, gamma=2.0):
        """Focal Loss实现，解决类别不平衡问题"""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        if alpha is not None:
            alpha = alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** gamma * ce_loss
            
        return focal_loss.mean()
    def forward(self, input_ids, attention_mask,labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state[:,0,:])  # 通常使用[CLS]位点表示整个片段
        logits = self.classifier(x)
        if labels is not None:
            #loss_fct = nn.CrossEntropyLoss()
            loss_fct = self.focal_loss
            # 注意：labels shape 应为 (B, L)，每个位置一个类别
            loss = loss_fct(logits.view(-1, logits.shape[-1]),labels,self.focal_alpha,self.focal_gamma)
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
                    #'pos': flank,  # 表示剪切点在序列中间
                    })
            all_preds.append(preds)
            all_labels.append(labels)
     # 拼接所有 batch 的预测与标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).sum().item() / len(all_labels)
    return acc,FP


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
        #    best_acc = acc1
        #    best_model_state = copy.deepcopy(model.state_dict())
        #    print(f"Epoch {epoch}: New best F1 = {acc1}, model saved.")
        #    if best_model_state is not None:
        #        torch.save({
        #            'model_state_dict': best_model_state,
        #            'best_acc': best_acc}
        #            ,save_path)
        #        print(f"Best model saved to {save_path}")
        #    else:
        #        print("No model was saved. Check training process.")
        
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
    main(True,True,train_num_samples=5000,validation_sample_num=50,sample_times=15,epochs=700,redirect=True)