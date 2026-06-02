import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import FullKVCache, StreamingKVCache, H2OKVCache, HybridKVCache, cacheSequenceLength


def appendNativeToken(pastKeyValues, tokenValue: float):
    newKey = torch.full((1, 2, 1, 4), tokenValue)
    newValue = torch.full((1, 2, 1, 4), tokenValue)
    if pastKeyValues is None:
        return ((newKey, newValue),)
    previousKey, previousValue = pastKeyValues[0]
    return ((
        torch.cat([previousKey, newKey], dim=-2),
        torch.cat([previousValue, newValue], dim=-2),
    ),)


def buildAttention(keyLength: int, preferredIndex: int = 0):
    attention = torch.full((1, 2, 1, keyLength), 0.01)
    attention[..., preferredIndex] = 1.0
    return (attention,)


def runPolicy(policy, steps: int = 10):
    pastKeyValues = None
    for step in range(steps):
        pastKeyValues = appendNativeToken(pastKeyValues, float(step))
        attentions = None
        if policy.requiresAttentions:
            preferredIndex = min(1, cacheSequenceLength(pastKeyValues) - 1)
            attentions = buildAttention(cacheSequenceLength(pastKeyValues), preferredIndex)
        pastKeyValues = policy.manage(pastKeyValues, attentions)
    return pastKeyValues


def testFullCache():
    cache = FullKVCache()
    pastKeyValues = runPolicy(cache, steps=10)
    assert cacheSequenceLength(pastKeyValues) == 10


def testStreamingCache():
    cache = StreamingKVCache(budget=5, sinkTokens=2)
    pastKeyValues = runPolicy(cache, steps=10)
    assert cacheSequenceLength(pastKeyValues) == 5
    assert cache.getCachedPositions() == [0, 1, 7, 8, 9]


def testH2OCache():
    cache = H2OKVCache(budget=5, sinkTokens=1, recentTokens=1)
    pastKeyValues = runPolicy(cache, steps=10)
    assert cacheSequenceLength(pastKeyValues) == 5
    assert 0 in cache.getCachedPositions()
    assert 9 in cache.getCachedPositions()


def testHybridCache():
    cache = HybridKVCache(budget=6, sinkTokens=1, recentTokens=2)
    pastKeyValues = runPolicy(cache, steps=10)
    assert cacheSequenceLength(pastKeyValues) == 6
    assert 0 in cache.getCachedPositions()
    assert 8 in cache.getCachedPositions()
    assert 9 in cache.getCachedPositions()


if __name__ == "__main__":
    testFullCache()
    testStreamingCache()
    testH2OCache()
    testHybridCache()
    print("All cache policy smoke tests passed.")
