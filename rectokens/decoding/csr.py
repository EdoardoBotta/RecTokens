from rectokens.decoding.trie import Trie
from collections import deque
from torch import sparse_csr_tensor

def csr_from_trie(trie: Trie):
    row_ptrs = []
    col_idxs = []
    
    frontier = deque([trie.root])
    while frontier:
        node = frontier.popleft()
        row_ptrs.append(len(col_idxs))
        for idx, child in node.children.items():
            col_idxs.append(idx)
            frontier.append(child)
    
    col_idxs.append(-1)
    values = [i for i in range(1, len(col_idxs))] + [-1]
    return sparse_csr_tensor(row_ptrs, col_idxs, values)


    

