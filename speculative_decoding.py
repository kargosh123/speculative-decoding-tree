import argparse

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from TreeNode import print_tree, Token, TreeNode


def sample(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Sample from the distribution to get the next token
    """
    return torch.topk(logits, top_k).indices


def evaluate_branches(
    root: TreeNode,
    prefix: torch.Tensor,
    mq_probs: torch.Tensor,
    Mp: torch.nn.Module,
    Mq: torch.nn.Module,
    gamma: int,
    level: int = 0,
):
    """
    Evaluate the tree and only sample from M_p when 
    particular branches match the condition provided in the paper

    Parameters:
    Mp: M_p distribution
    Mq: M_q distribution
    level: number of nodes in the tree
    prefix: encoded tokens
    root: root node of the tree to start
    gamma: number of tokens to generate

    Returns:
    longest_prefix: encoded tokens that satisfy the speculative condition
    """
    if level >= gamma:
        # Get the distribution of the last gamma + 1 in the sequence
        # p: [1, gamma + 1, vocab_size]

        p = F.softmax(Mp(prefix).logits, dim=-1)[:, -gamma - 1 :]
        # Ignore the last token in the sequence for now (only used if all tokens
        # are accepted). mp_probs: [1, gamma, 1]

        mp_probs = p[:, :-1].gather(2, prefix[:, -gamma:].unsqueeze(0)).squeeze(0)

        div = mp_probs / mq_probs
        r = torch.rand_like(div)
        comparison = (r > div).type(torch.int64).squeeze()
        first_reject = comparison.argmax()
        if first_reject == 0 and comparison[0] == 0:
            full_cat = torch.cat((prefix, p[:, -1].argmax().reshape((1, 1))), dim=-1)

            return full_cat
        else:
            curr = root
            for _ in range(gamma - first_reject):
                curr = curr.parent
            p_prime = p[:, -(gamma - first_reject)] - curr.val.probs
            p_prime = p_prime.clamp_min(0)
            p_prime = p_prime / p_prime.sum()
            partial_cat = torch.cat(
                (
                    prefix[:, : -(gamma - first_reject)],
                    p_prime.argmax().reshape((1, 1)),
                ),
                dim=-1,
            )

            return partial_cat
    longest_prefix = torch.Tensor([[]])
    for child in root.children:
        curr_prefix = evaluate_branches(
            child,
            torch.cat((prefix, child.val.index.reshape((1, 1))), dim=-1),
            torch.cat((mq_probs, child.val.get_prob().reshape((1, 1))), dim=-1),
            Mp,
            Mq,
            gamma,
            level + 1,
        )
        if len(curr_prefix[0]) > len(longest_prefix[0]):
            longest_prefix = curr_prefix
    return longest_prefix


def build_tree_v2(
    root: TreeNode,
    prefix: torch.Tensor,
    Mq: torch.nn.Module,
    gamma: int,
    top_k: int,
    curr_level: int = 0,
):
    """
    Build an encoded token tree based on generated samples from M_q
    This occurs recursively, in-place

    Parameters:
    root: root node of the tree to start
    prefix: encoded tokens
    Mq: M_q distribution
    gamma: number of tokens to generate
    top_k: number of samples to generate
    """
    if curr_level >= gamma:
        return
    qi: torch.Tensor = Mq(prefix).logits
    next_children_tokens = sample(qi[:, -1, :], top_k)
    for i in range(top_k):
        token_idx = next_children_tokens[:, i].squeeze()
        token_probs = F.softmax(qi[:, -1, :]).squeeze()
        token = Token(probs=token_probs, index=token_idx)
        child = TreeNode(token)
        root.add_child(child)
        build_tree_v2(
            child,
            torch.cat((prefix, child.val.index.reshape((1, 1))), dim=1),
            Mq,
            gamma,
            top_k,
            curr_level=curr_level + 1,
        )


def speculative_decoding_step(
    Mp: torch.nn.Module,
    Mq: torch.nn.Module,
    prefix: torch.Tensor,
    gamma=4,
    top_k=4,
) -> torch.Tensor:
    """
    Based on the Leviathan paper, we implement algorithm 1 using a k-ary tree based approach
    instead of the linear approach described in the paper. We maintain the same acceptance
    condition as in the paper as in the end we still want to determine if our small model sample
    provides a sufficiently similar distribution to the larger target model.

    Parameters:
    Mp: M_p distribution
    Mq: M_q distribution
    prefix: encoded tokens
    gamma: number of tokens to generate
    top_k: number of tokens to sample from M_q to build the tree
    """
    # For every desired token, we generate a gamma*top_k tree

    probabilities = torch.Tensor([[]])
    xinit = TreeNode(Token(index=prefix, probs=torch.Tensor([[1.0]])))

    build_tree_v2(xinit, prefix, Mq, gamma, top_k)
    print("Tree: ")
    print_tree(xinit)

    # Determine the number of accepted guesses

    prefix = evaluate_branches(xinit, prefix, probabilities, Mp, Mq, gamma)

    return prefix


parser = argparse.ArgumentParser(description="Run the tree-based speculative decoding")
parser.add_argument("--input", type=str, help="The input text to generate from")
parser.add_argument(
    "--Mp", type=str, help="The path to the Mp model", default="bigscience/bloom-1b1"
)
parser.add_argument(
    "--Mq", type=str, help="The path to the Mq model", default="bigscience/bloom-560m"
)
parser.add_argument(
    "--gamma", type=int, help="The number of tokens to generate", default=2
)
parser.add_argument(
    "--top_k",
    type=int,
    help="The number of tokens to sample at every node of the tree",
    default=4,
)
parser.add_argument(
    "--rounds",
    type=int,
    help="How many rounds of speculative decoding to run",
    default=2,
)

if __name__ == "__main__":
    args = parser.parse_args()

    input_text = args.input
    tokenizer = AutoTokenizer.from_pretrained(args.Mp)

    Mp = AutoModelForCausalLM.from_pretrained(args.Mp)
    Mq = AutoModelForCausalLM.from_pretrained(args.Mq)
    print(f"input: {input_text}")
    input_tokens = torch.Tensor(tokenizer.encode(input_text, return_tensors="pt"))
    for i in range(args.rounds):
        output = speculative_decoding_step(
            Mp, Mq, prefix=input_tokens, gamma=args.gamma, top_k=args.top_k
        )
        input_tokens = output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Round {i}: {generated_text}")
