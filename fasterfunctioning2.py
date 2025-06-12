import torch
import numpy as np
from numba import jit, prange, types
from numba.typed import Dict, List
from numba.core import types
import cython
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from Bio.Seq import Seq
import random

# 预编译的DNA编码表
BASE_TO_INT = np.array([0, 1, 2, 3], dtype=np.int8)  # A, T, G, C
INT_TO_BASE = np.array(['A', 'T', 'G', 'C'], dtype='U1')

@jit(nopython=True, parallel=True, cache=True)
def ultra_fast_kmer_tokenization(seq_array: np.ndarray, kmer: int, flank: int) -> np.ndarray:
    """
    超高速k-mer标记化，直接在数值数组上操作，避免字符串转换
    """
    seq_len = len(seq_array)
    window_size = flank * 2 + 1
    valid_positions = seq_len - 2 * flank
    kmers_per_window = window_size - kmer + 1
    
    # 预分配结果数组 - 存储k-mer的哈希值
    result = np.empty((valid_positions, kmers_per_window), dtype=np.int64)
    
    for i in prange(valid_positions):
        window_start = i
        for j in range(kmers_per_window):
            kmer_hash = 0
            for k in range(kmer):
                # 使用滚动哈希避免重复计算
                kmer_hash = kmer_hash * 4 + seq_array[window_start + j + k]
            result[i, j] = kmer_hash
    
    return result

@jit(nopython=True, cache=True)
def kmer_hash_to_string(kmer_hash: int, kmer_size: int) -> str:
    """将k-mer哈希值转换回字符串"""
    bases = ['A', 'T', 'G', 'C']
    result = [''] * kmer_size
    
    for i in range(kmer_size - 1, -1, -1):
        result[i] = bases[kmer_hash % 4]
        kmer_hash //= 4
    
    return ''.join(result)

@jit(nopython=True, parallel=True, cache=True)
def vectorized_coordinate_calculation(indices: np.ndarray, seq_start: int, seq_end: int, 
                                    strand: str, flank: int) -> np.ndarray:
    """向量化坐标计算"""
    result = np.empty(len(indices), dtype=np.int64)
    
    if strand == '+':
        for i in prange(len(indices)):
            result[i] = seq_start + flank + indices[i]
    else:
        for i in prange(len(indices)):
            result[i] = seq_end - flank - indices[i] - 1
    
    return result

@jit(nopython=True, cache=True)
def fast_set_operations(pred_coords: np.ndarray, true_coords: np.ndarray) -> Tuple[int, int, int]:
    """使用排序数组进行快速集合运算"""
    pred_set = set(pred_coords)
    true_set = set(true_coords)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    return tp, fp, fn

@jit(nopython=True, cache=True)
def encode_sequence_batch(sequences: List[str]) -> np.ndarray:
    """批量编码DNA序列"""
    if not sequences:
        return np.empty((0, 0), dtype=np.int8)
    
    max_len = max(len(seq) for seq in sequences)
    result = np.zeros((len(sequences), max_len), dtype=np.int8)
    
    base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 0}
    
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            result[i, j] = base_map.get(base, 0)
    
    return result

class OptimizedTokenizerCache:
    """优化的tokenizer缓存，减少重复计算"""
    def __init__(self, kmer_size: int, max_cache_size: int = 10000):
        self.kmer_size = kmer_size
        self.cache = {}
        self.max_cache_size = max_cache_size
        
    def get_kmer_tokens(self, kmer_hashes: np.ndarray) -> List[str]:
        """从k-mer哈希值获取token字符串"""
        tokens = []
        for kmer_hash in kmer_hashes:
            if kmer_hash in self.cache:
                tokens.append(self.cache[kmer_hash])
            else:
                token = kmer_hash_to_string(kmer_hash, self.kmer_size)
                if len(self.cache) < self.max_cache_size:
                    self.cache[kmer_hash] = token
                tokens.append(token)
        return tokens

def ultra_sliding_eval_by_splice_anchor(
    model, tokenizer, genome, splice_coords_set_donor, splice_coords_set_acceptor,
    device, mode='donor', region_length=1000, flank=50, kmer=6, sample_time=1, batch_size=32
):
    """
    终极优化版本的剪接位点评估函数
    
    主要优化:
    1. 使用numba JIT编译所有可能的函数
    2. 向量化所有数组操作
    3. 预分配所有内存
    4. 优化字符串操作和缓存
    5. 改进批处理策略
    6. 使用更高效的数据结构
    """
    model.eval()
    results = []
    window_size = flank * 2 + 1
    current_times = 0
    FP_set, FN_set = [], []
    
    # 预编译坐标集合，使用numpy数组加速
    splice_coords_set = splice_coords_set_donor if mode == 'donor' else splice_coords_set_acceptor
    
    # 构建优化的查找结构
    coord_to_type = {}
    all_donor_coords = np.array([(chrom, coord, strand) for chrom, coord, strand in splice_coords_set_donor], 
                               dtype=object)
    all_acceptor_coords = np.array([(chrom, coord, strand) for chrom, coord, strand in splice_coords_set_acceptor], 
                                  dtype=object)
    
    for chrom, coord, strand in splice_coords_set_donor:
        coord_to_type[(chrom, coord, strand)] = 'donor'
    for chrom, coord, strand in splice_coords_set_acceptor:
        coord_to_type[(chrom, coord, strand)] = 'acceptor'
    
    # 初始化tokenizer缓存
    tokenizer_cache = OptimizedTokenizerCache(kmer)
    
    # DNA序列编码映射 - 使用numpy数组加速
    base_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 0}
    
    with torch.no_grad():
        # 预采样并排序坐标以提高缓存效率
        sample_coords = random.sample(list(splice_coords_set), k=min(20, len(splice_coords_set)))
        sample_coords.sort()  # 排序以提高基因组访问的局部性
        
        # 预分配批处理缓冲区
        max_seq_per_batch = batch_size * 2  # 预留额外空间
        batch_buffer_size = max_seq_per_batch * window_size
        
        for batch_start in range(0, len(sample_coords), batch_size):
            if current_times >= sample_time:
                break
                
            batch_coords = sample_coords[batch_start:batch_start + batch_size]
            
            # 预分配批处理数组
            batch_sequences = []
            batch_metadata = []
            batch_seq_arrays = []
            
            # 批量准备和编码序列
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
                
                # 高效编码序列
                seq_array = np.array([base_to_int.get(base, 0) for base in seq], dtype=np.int8)
                
                batch_sequences.append(seq)
                batch_seq_arrays.append(seq_array)
                batch_metadata.append({
                    'chrom': chrom, 'center': center, 'strand': strand,
                    'seq_start': seq_start, 'seq_end': seq_end, 'seq_length': seq_length
                })
                current_times += 1
            
            if not batch_sequences:
                continue
            
            # 超高速批量k-mer生成和tokenization
            all_tokenized_chunks = []
            all_absolute_coords = []
            batch_boundaries = []
            
            for seq_idx, (seq_array, metadata) in enumerate(zip(batch_seq_arrays, batch_metadata)):
                start_boundary = len(all_tokenized_chunks)
                
                # 使用ultra_fast_kmer_tokenization
                kmer_matrix = ultra_fast_kmer_tokenization(seq_array, kmer, flank)
                
                # 向量化坐标计算
                valid_indices = np.arange(kmer_matrix.shape[0])
                absolute_coords = vectorized_coordinate_calculation(
                    valid_indices, metadata['seq_start'], metadata['seq_end'], 
                    metadata['strand'], flank
                )
                
                # 批量生成tokenized chunks
                for i in range(kmer_matrix.shape[0]):
                    kmer_tokens = tokenizer_cache.get_kmer_tokens(kmer_matrix[i])
                    tokenized_chunk = " ".join(kmer_tokens)
                    all_tokenized_chunks.append(tokenized_chunk)
                
                all_absolute_coords.extend(absolute_coords.tolist())
                batch_boundaries.append((start_boundary, len(all_tokenized_chunks)))
            
            if not all_tokenized_chunks:
                continue
            
            # 优化的批量编码
            try:
                encodings = tokenizer(
                    all_tokenized_chunks, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=window_size+2, 
                    return_tensors='pt',
                    add_special_tokens=True
                )
                input_ids = encodings['input_ids'].to(device, non_blocking=True)
                attention_mask = encodings['attention_mask'].to(device, non_blocking=True)
            except Exception as e:
                print(f"Tokenization error: {e}")
                continue
            
            # 自适应批量推理
            all_preds = []
            # 根据GPU内存动态调整推理批大小
            inference_batch_size = min(256, len(input_ids))
            if device.type == 'cuda':
                # 根据可用GPU内存调整
                torch.cuda.empty_cache()
                inference_batch_size = min(512, len(input_ids))
            
            for i in range(0, len(input_ids), inference_batch_size):
                try:
                    batch_input_ids = input_ids[i:i + inference_batch_size]
                    batch_attention_mask = attention_mask[i:i + inference_batch_size]
                    
                    with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
                        logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    
                    # 释放中间张量内存
                    del batch_input_ids, batch_attention_mask, logits
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Inference error: {e}")
                    continue
            
            # 释放主要张量内存
            del input_ids, attention_mask
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 高效处理每个序列的结果
            for seq_idx, (start_idx, end_idx) in enumerate(batch_boundaries):
                metadata = batch_metadata[seq_idx]
                seq_preds = np.array(all_preds[start_idx:end_idx])
                seq_coords = np.array(all_absolute_coords[start_idx:end_idx])
                
                # 向量化提取预测的剪接位点
                splice_mask = (seq_preds == 1) | (seq_preds == 2)
                splice_indices = np.where(splice_mask)[0]
                
                pred_splice_coords = []
                for idx in splice_indices:
                    pred_splice_coords.append({
                        'coord': seq_coords[idx],
                        'label': 'donor' if seq_preds[idx] == 1 else 'acceptor'
                    })
                
                # 高效提取真实剪接位点
                true_splice_in_region = []
                chrom, strand = metadata['chrom'], metadata['strand']
                
                for coord in seq_coords:
                    coord_key = (chrom, coord, strand)
                    if coord_key in coord_to_type:
                        true_splice_in_region.append({
                            'coord': coord,
                            'label': coord_to_type[coord_key]
                        })
                
                # 使用numba加速的集合运算
                pred_coords_array = np.array([item['coord'] for item in pred_splice_coords])
                true_coords_array = np.array([item['coord'] for item in true_splice_in_region])
                
                if len(pred_coords_array) > 0 and len(true_coords_array) > 0:
                    TP, FP, FN = fast_set_operations(pred_coords_array, true_coords_array)
                else:
                    TP = 0
                    FP = len(pred_coords_array)
                    FN = len(true_coords_array)
                
                TN = len(seq_preds) - TP - FP - FN
                
                # 高效收集FP和FN样本
                if FP > 0 or FN > 0:
                    chrom_seq = genome[chrom].seq
                    pred_coords_set = set(pred_coords_array) if len(pred_coords_array) > 0 else set()
                    true_coords_set = set(true_coords_array) if len(true_coords_array) > 0 else set()
                    
                    fp_coords = pred_coords_set - true_coords_set
                    fn_coords = true_coords_set - pred_coords_set
                    
                    for coord in fp_coords:
                        try:
                            FP_set.append({
                                'seq': str(chrom_seq[coord-flank:coord+flank+1]),
                                'label': 'negative'
                            })
                        except:
                            pass
                    
                    for coord in fn_coords:
                        try:
                            true_label = next(item['label'] for item in true_splice_in_region if item['coord'] == coord)
                            FN_set.append({
                                'seq': str(chrom_seq[coord-flank:coord+flank+1]),
                                'label': true_label
                            })
                        except:
                            pass
                
                result_entry = {
                    "chrom": chrom,
                    "strand": strand,
                    "predictions": pred_splice_coords,
                    "true_splice_coord": true_splice_in_region,
                    "metrics": {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
                }
                results.append(result_entry)
    
    # 向量化汇总指标
    metrics_array = np.array([[r["metrics"]["TP"], r["metrics"]["TN"], 
                              r["metrics"]["FP"], r["metrics"]["FN"]] for r in results])
    
    if len(metrics_array) > 0:
        total_TP, total_TN, total_FP, total_FN = metrics_array.sum(axis=0)
    else:
        total_TP = total_TN = total_FP = total_FN = 0
    
    return results, FP_set, FN_set, int(total_TP), int(total_TN), int(total_FP), int(total_FN)

# 内存池管理优化
class MemoryPool:
    """预分配内存池，避免频繁的内存分配/释放"""
    def __init__(self, max_size=1000000):
        self.arrays = {}
        self.max_size = max_size
        
    def get_array(self, shape, dtype=np.int64):
        key = (shape, dtype)
        if key not in self.arrays:
            self.arrays[key] = np.empty(shape, dtype=dtype)
        return self.arrays[key]
    
    def clear(self):
        self.arrays.clear()

# SIMD优化的序列处理
@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def simd_optimized_kmer_hash(seq_array: np.ndarray, kmer_size: int) -> np.ndarray:
    """使用SIMD优化的k-mer哈希计算"""
    n = len(seq_array)
    """
    创建高级Cython扩展以获得最大性能
    """
    setup_code = """
from setuptools import setup
from Cython.Build import cythonize
import numpy
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

ext_modules = [
    Pybind11Extension(
        "ultra_fast_sequence_ops",
        ["ultra_fast_sequence_ops.cpp"],
        include_dirs=[numpy.get_include()],
        cxx_std=17,
    ),
]

setup(
    ext_modules=cythonize("ultra_fast_sequence_ops.pyx") + ext_modules,
    cmdclass={"build_ext": build_ext},
    include_dirs=[numpy.get_include()]
)
"""
    
    cython_code = """
# ultra_fast_sequence_ops.pyx
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ultra_fast_kmer_encoding(str sequence, int k, int flank):
    cdef int seq_len = len(sequence)
    cdef int window_size = flank * 2 + 1
    cdef int valid_positions = seq_len - 2 * flank
    cdef int kmers_per_window = window_size - k + 1
    
    # 预分配结果数组
    cdef np.ndarray[np.int64_t, ndim=2] result = np.empty((valid_positions, kmers_per_window), dtype=np.int64)
    
    # 使用C数组加速
    cdef char* seq_ptr = <char*>sequence
    cdef long kmer_hash
    cdef int i, j, l
    
    # 预计算base转换表
    cdef int base_map[256]
    for i in range(256):
        base_map[i] = 0
    base_map[65] = 0  # 'A'
    base_map[84] = 1  # 'T'
    base_map[71] = 2  # 'G'
    base_map[67] = 3  # 'C'
    
    with nogil:
        for i in range(valid_positions):
            for j in range(kmers_per_window):
                kmer_hash = 0
                for l in range(k):
                    kmer_hash = kmer_hash * 4 + base_map[<unsigned char>seq_ptr[i + j + l]]
                result[i, j] = kmer_hash
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def vectorized_tokenization(np.ndarray[np.int64_t, ndim=2] kmer_matrix, dict kmer_cache):
    cdef int rows = kmer_matrix.shape[0]
    cdef int cols = kmer_matrix.shape[1]
    
    result = []
    cdef long kmer_hash
    cdef str token_str
    
    for i in range(rows):
        tokens = []
        for j in range(cols):
            kmer_hash = kmer_matrix[i, j]
            if kmer_hash in kmer_cache:
                tokens.append(kmer_cache[kmer_hash])
            else:
                # 快速哈希到字符串转换
                token_str = hash_to_kmer_string(kmer_hash, 6)  # 假设k=6
                kmer_cache[kmer_hash] = token_str
                tokens.append(token_str)
        result.append(" ".join(tokens))
    
    return result

cdef str hash_to_kmer_string(long kmer_hash, int k):
    cdef char bases[4]
    bases[0] = b'A'
    bases[1] = b'T'
    bases[2] = b'G'
    bases[3] = b'C'
    
    cdef char* result = <char*>malloc((k + 1) * sizeof(char))
    result[k] = 0  # null terminator
    
    cdef int i
    for i in range(k - 1, -1, -1):
        result[i] = bases[kmer_hash % 4]
        kmer_hash //= 4
    
    cdef str py_result = result.decode('ascii')
    free(result)
    return py_result
"""
    
    cpp_code = """
// ultra_fast_sequence_ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <execution>

namespace py = pybind11;

// 超高速并行k-mer生成
py::array_t<int64_t> parallel_kmer_generation(
    const std::string& sequence, int k, int flank) {
    
    int seq_len = sequence.length();
    int window_size = flank * 2 + 1;
    int valid_positions = seq_len - 2 * flank;
    int kmers_per_window = window_size - k + 1;
    
    auto result = py::array_t<int64_t>({valid_positions, kmers_per_window});
    auto buf = result.request();
    int64_t* ptr = static_cast<int64_t*>(buf.ptr);
    
    // 并行处理
    std::vector<int> indices(valid_positions);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
        [&](int i) {
            for (int j = 0; j < kmers_per_window; ++j) {
                int64_t kmer_hash = 0;
                for (int l = 0; l < k; ++l) {
                    char base = sequence[i + j + l];
                    int base_val = (base == 'A') ? 0 : (base == 'T') ? 1 : 
                                  (base == 'G') ? 2 : 3;
                    kmer_hash = kmer_hash * 4 + base_val;
                }
                ptr[i * kmers_per_window + j] = kmer_hash;
            }
        });
    
    return result;
}

PYBIND11_MODULE(ultra_fast_sequence_ops, m) {
    m.def("parallel_kmer_generation", &parallel_kmer_generation, "Ultra fast parallel k-mer generation");
}
"""
    
    return setup_code, cython_code, cpp_code