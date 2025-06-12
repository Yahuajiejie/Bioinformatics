import torch
import numpy as np
from numba import jit, prange
import cython
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from Bio.Seq import Seq
import random
# Cython加速的序列处理函数
@jit(nopython=True, parallel=True)
def fast_kmer_generation(seq_array: np.ndarray, kmer: int, flank: int) -> np.ndarray:
    """使用numba加速k-mer生成"""
    seq_len = len(seq_array)
    window_size = flank * 2 + 1
    valid_positions = seq_len - 2 * flank
    
    # 预分配结果数组
    kmers_per_window = window_size - kmer + 1
    result = np.empty((valid_positions, kmers_per_window), dtype=np.int32)
    
    for i in prange(valid_positions):
        start_pos = i
        end_pos = i + window_size
        for j in range(kmers_per_window):
            kmer_start = start_pos + j
            # 将k-mer编码为整数（简化版本）
            kmer_val = 0
            for k in range(kmer):
                kmer_val = kmer_val * 4 + seq_array[kmer_start + k]
            result[i, j] = kmer_val
    
    return result

@jit(nopython=True)
def fast_coordinate_matching(pred_coords: np.ndarray, true_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """快速坐标匹配，返回TP和FP的索引"""
    tp_indices = []
    fp_indices = []
    
    # 将true_coords转换为集合以快速查找
    true_set = set(true_coords)
    
    for i, coord in enumerate(pred_coords):
        if coord in true_set:
            tp_indices.append(i)
        else:
            fp_indices.append(i)
    
    return np.array(tp_indices), np.array(fp_indices)

def sliding_eval_by_splice_anchor_optimized(
    model, tokenizer, genome, splice_coords_set_donor, splice_coords_set_acceptor,
    device, mode='donor', region_length=1000, flank=50, kmer=6, sample_time=1, batch_size=32
):
    """优化版本的剪接位点评估函数"""
    model.eval()
    results = []
    window_size = flank * 2 + 1
    current_times = 0
    FP_set, FN_set = [], []
    
    # 预编译坐标集合为numpy数组和字典，加速查找
    splice_coords_set = splice_coords_set_donor if mode == 'donor' else splice_coords_set_acceptor
    coord_to_type = {}
    
    # 构建快速查找字典
    for chrom, coord, strand in splice_coords_set_donor:
        coord_to_type[(chrom, coord, strand)] = 'donor'
    for chrom, coord, strand in splice_coords_set_acceptor:
        coord_to_type[(chrom, coord, strand)] = 'acceptor'
    
    # DNA序列编码映射
    base_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 0}
    
    with torch.no_grad():
        # 批量处理采样的序列
        sample_coords = random.sample(list(splice_coords_set), k=min(20, len(splice_coords_set)))
        
        for batch_start in range(0, len(sample_coords), batch_size):
            if current_times >= sample_time:
                break
                
            batch_coords = sample_coords[batch_start:batch_start + batch_size]
            batch_sequences = []
            batch_metadata = []
            
            # 批量准备序列
            for chrom, center, strand in batch_coords:
                if current_times >= sample_time:
                    break
                    
                chrom_seq = genome[chrom].seq
                region_length_pre = random.randint(0, region_length // 2)
                region_length_post = region_length - region_length_pre
                seq_start = center - region_length_pre - flank
                seq_end = center + region_length_post + flank
                
                if seq_start < 0 or seq_end >= len(chrom_seq):
                    continue
                    
                seq_length = seq_end - seq_start
                seq = str(chrom_seq[seq_start:seq_end])
                if strand == '-':
                    seq = str(Seq(seq).reverse_complement())
                
                # 将序列转换为数字数组以便加速处理
                seq_array = np.array([base_to_int.get(base, 0) for base in seq], dtype=np.int32)
                
                batch_sequences.append(seq_array)
                batch_metadata.append({
                    'chrom': chrom, 'center': center, 'strand': strand,
                    'seq_start': seq_start, 'seq_end': seq_end, 'seq_length': seq_length
                })
                current_times += 1
            
            # 批量处理k-mer生成和tokenization
            all_tokenized_chunks = []
            all_absolute_coords = []
            batch_boundaries = []
            
            for seq_idx, (seq_array, metadata) in enumerate(zip(batch_sequences, batch_metadata)):
                # 使用numba加速的k-mer生成
                start_boundary = len(all_tokenized_chunks)
                
                absolute_coords = []
                tokenized_chunks = []
                
                for i in range(flank, metadata['seq_length'] - flank):
                    window_seq = ''.join([['A','T','G','C'][base] for base in seq_array[i-flank:i+flank+1]])
                    kmer_seq = " ".join([window_seq[j:j + kmer] for j in range(len(window_seq) - kmer + 1)])
                    tokenized_chunks.append(kmer_seq)
                    
                    if metadata['strand'] == '+':
                        coord = metadata['seq_start'] + i
                    else:
                        coord = metadata['seq_end'] - i - 1
                    absolute_coords.append(coord)
                
                all_tokenized_chunks.extend(tokenized_chunks)
                all_absolute_coords.extend(absolute_coords)
                batch_boundaries.append((start_boundary, len(all_tokenized_chunks)))
            
            if not all_tokenized_chunks:
                continue
            
            # 批量编码 - 一次性处理所有序列
            encodings = tokenizer(
                all_tokenized_chunks, 
                padding='max_length', 
                truncation=True, 
                max_length=window_size, 
                return_tensors='pt'
            )
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # 分批推理以避免内存溢出
            all_preds = []
            inference_batch_size = 256
            
            for i in range(0, len(input_ids), inference_batch_size):
                batch_input_ids = input_ids[i:i + inference_batch_size]
                batch_attention_mask = attention_mask[i:i + inference_batch_size]
                
                logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                # we are using model2
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
            
            # 处理每个序列的结果
            for seq_idx, (start_idx, end_idx) in enumerate(batch_boundaries):
                metadata = batch_metadata[seq_idx]
                seq_preds = all_preds[start_idx:end_idx]
                seq_coords = all_absolute_coords[start_idx:end_idx]
                
                # 提取预测的剪接位点
                pred_splice_coords = []
                for i, p in enumerate(seq_preds):
                    if p in (1, 2):
                        pred_splice_coords.append({
                            'coord': seq_coords[i],
                            'label': 'donor' if p == 1 else 'acceptor'
                        })
                
                # 快速提取真实剪接位点
                true_splice_in_region = []
                chrom, strand = metadata['chrom'], metadata['strand']
                
                for coord in seq_coords:
                    coord_key = (chrom, coord, strand)
                    if coord_key in coord_to_type:
                        true_splice_in_region.append({
                            'coord': coord,
                            'label': coord_to_type[coord_key]
                        })
                
                # 使用集合快速计算TP, FP, FN
                pred_coords_set = {item['coord'] for item in pred_splice_coords}
                true_coords_set = {item['coord'] for item in true_splice_in_region}
                
                tp_coords = pred_coords_set & true_coords_set
                fp_coords = pred_coords_set - true_coords_set
                fn_coords = true_coords_set - pred_coords_set
                
                TP = len(tp_coords)
                FP = len(fp_coords)
                FN = len(fn_coords)
                TN = len(seq_preds) - TP - FP - FN
                
                # 收集FP和FN样本
                chrom_seq = genome[chrom].seq
                for coord in fp_coords:
                    FP_set.append({
                        'seq': str(chrom_seq[coord-flank:coord+flank+1]),
                        'label': 'negative'
                    })
                
                for coord in fn_coords:
                    true_label = next(item['label'] for item in true_splice_in_region if item['coord'] == coord)
                    FN_set.append({
                        'seq': str(chrom_seq[coord-flank:coord+flank+1]),
                        'label': true_label
                    })
                
                result_entry = {
                    "chrom": chrom,
                    "strand": strand,
                    "predictions": pred_splice_coords,
                    "true_splice_coord": true_splice_in_region,
                    "metrics": {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
                }
                results.append(result_entry)
    
    # 汇总所有批次的指标
    total_TP = sum(r["metrics"]["TP"] for r in results)
    total_TN = sum(r["metrics"]["TN"] for r in results)
    total_FP = sum(r["metrics"]["FP"] for r in results)
    total_FN = sum(r["metrics"]["FN"] for r in results)
    
    return results, FP_set, FN_set, total_TP, total_TN, total_FP, total_FN

# 额外的Cython优化函数（需要单独编译）
def create_cython_extension():
    """
    创建Cython扩展的设置脚本
    保存为setup.py并运行: python setup.py build_ext --inplace
    """
    setup_code = """
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("fast_sequence_ops.pyx"),
    include_dirs=[numpy.get_include()]
)
"""
    
    cython_code = """
# fast_sequence_ops.pyx
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_kmer_encode(str sequence, int k):
    cdef int seq_len = len(sequence)
    cdef int num_kmers = seq_len - k + 1
    cdef np.ndarray[np.int64_t, ndim=1] result = np.empty(num_kmers, dtype=np.int64)
    cdef int i, j
    cdef long kmer_hash
    
    for i in range(num_kmers):
        kmer_hash = 0
        for j in range(k):
            if sequence[i + j] == 'A':
                kmer_hash = kmer_hash * 4 + 0
            elif sequence[i + j] == 'T':
                kmer_hash = kmer_hash * 4 + 1
            elif sequence[i + j] == 'G':
                kmer_hash = kmer_hash * 4 + 2
            else:  # C or other
                kmer_hash = kmer_hash * 4 + 3
        result[i] = kmer_hash
    
    return result
"""
    
    return setup_code, cython_code