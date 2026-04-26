TCGA_PARAMS = {
    "immune_hot": {
        "inspired_by": "Skin Cutaneous Melanoma (SKCM), n=440",
        "tmb": {"dist": "lognormal", "mean_log": 2.5998, "std_log": 1.2026},
        "genomic_instability": {"dist": "normal", "mean": 0.3218, "std": 0.2100},
        "mutation_count": {"dist": "lognormal", "mean_log": 5.9979, "std_log": 1.1969},
    },
    "immune_cold": {
        "inspired_by": "Glioblastoma Multiforme (GBM), n=397",
        "tmb": {"dist": "lognormal", "mean_log": 0.6189, "std_log": 0.7192},
        "genomic_instability": {"dist": "normal", "mean": 0.2071, "std": 0.1221},
        "mutation_count": {"dist": "lognormal", "mean_log": 4.0258, "std_log": 0.6904},
    },
    "high_mutation": {
        "inspired_by": "Top 25% TMB cohort from SKCM, n=110",
        "tmb": {"dist": "lognormal", "mean_log": 6.5080, "std_log": 0.8088},
        "genomic_instability": {"dist": "normal", "mean": 0.3218, "std": 0.2100},
        "mutation_count": {"dist": "lognormal", "mean_log": 4.0080, "std_log": 0.5068},
    },
}
