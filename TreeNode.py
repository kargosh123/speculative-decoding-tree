import attrs
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")


@attrs.frozen
class Token:
    # The probability distribution of the current path. [1, vocab_size]
    probs: torch.Tensor
    # The index of the token in the vocabulary
    index: torch.Tensor

    def __str__(self):
        """Get the string representation of the token"""
        token_str = (
            tokenizer.decode(self.index.squeeze())
            .encode("unicode_escape")
            .decode("utf-8")
            .replace(" ", "_")
        )
        return f"Token(prob={self.get_prob()}, index={self.index}, str={token_str})"

    def get_prob(self):
        """Get the probability of the token from its distribution"""
        if len(self.index.shape) > 1:
            return 1.0
        return self.probs[self.index]


class TreeNode:
    def __init__(self, value: Token = None, parent: "TreeNode" = None):
        """
        TreeNode class to store encoded tokens and the generated sequences as a k-ary tree.
        Eventually could allow both backtracking and prefix sampling.
        """
        self.val = value
        self.children: list[TreeNode] = []
        self.parent = parent

    def add_child(self, child: "TreeNode"):
        """
        Add a child node to the current node
        """
        child.parent = self
        self.children.append(child)

    def __str__(self):
        return str(self.val)


def print_tree(node: TreeNode, depth=0):
    """
    Display the tree
    """
    print("\t" * depth, node)
    for child in node.children:
        print_tree(child, depth + 1)
