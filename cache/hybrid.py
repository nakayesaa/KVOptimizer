from .h2o import H2OKVCache


class HybridKVCache(H2OKVCache):

    cacheType = "hybrid"

    def __init__(
        self,
        budget: int,
        sinkTokens: int = 4,
        recentTokens: int = 32,
    ):
        super().__init__(
            budget=budget,
            sinkTokens=sinkTokens,
            recentTokens=recentTokens,
        )
