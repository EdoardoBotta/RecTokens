from __future__ import annotations

from typing import Optional


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_token = False
    
    def take_step(self, idx: int) -> Optional[TrieNode]:
        return self.children.get(idx)

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token: list[int]) -> None:
        node = self.root
        for idx in token:
            if idx not in node.children:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        node.is_end_of_token = True

    def find_prefix(self, token: list[int]) -> Optional[TrieNode]:
        node = self.root
        for idx in token:
            if idx not in node.children:
                return None
            node = node.children[idx]
        return node

