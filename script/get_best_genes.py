import pandas as pd
import os
import sys
sys.path.insert(0, '..')

cmmn = pd.read_csv("../evaluation/predictions/cmmn/predictions.csv")
hvg = pd.read_csv("../evaluation/predictions/hvg/predictions.csv")
icms2up = pd.read_csv("../evaluation/predictions/icms2up/predictions.csv")
icms2down = pd.read_csv("../evaluation/predictions/icms2down/predictions.csv")
icms3up = pd.read_csv("../evaluation/predictions/icms3up/predictions.csv")
icms3down = pd.read_csv("../evaluation/predictions/icms3down/predictions.csv")

dfs = {
    "cmmn": cmmn,
    "hvg": hvg,
    "icms2up": icms2up,
    "icms2down": icms2down,
    "icms3up": icms3up,
    "icms3down": icms3down,
}

for name, df in dfs.items():
    df["gene"] = df["gene"].str.replace("_", "", regex=False)

def print_best_pearson(df, top_k=50):
    best_rows = df.nlargest(top_k*20, "pearson")
    seen = set()
    best_pearson = []
    best_gene = []

    for _, row in best_rows.iterrows():
        gene = row["gene"]
        if gene in seen:
            continue
        seen.add(gene)

        best_pearson.append(row["pearson"])
        best_gene.append(gene)

        # print only the top_k unique genes
        print(row["pearson"], gene)
        if len(best_gene) >= top_k:
            break

    return pd.Series(best_pearson), pd.Series(best_gene)


best_pearsons = []
best_genes = []
for name, df in dfs.items():
    print("=== dataset:", name, "===")
    print_best_pearson(df)

    best_pearson, best_gene = print_best_pearson(df)
    best_pearsons.append(best_pearson)
    best_genes.append(best_gene)

print(best_pearsons)

cmmn_best = best_genes[0]
hvg_best = best_genes[1]
icms2up_best = best_genes[2]
icms2down_best = best_genes[3]
icms3up_best = best_genes[4]
icms3down_best = best_genes[5]

def check_cmmn_vs_hvg(cmmn_best, hvg_best):
    found = []
    not_found = []

    for i in cmmn_best:
        if i in hvg_best.values:
            found.append(i)
        else:
            not_found.append(i)

    print(found)

    print(not_found)

    print(len(found), len(not_found))

    print(cmmn_best, hvg_best)


def check_best_n(geneset, genes, pearsons, max_item=5):
    print("=== dataset:", geneset, "===")
    for idx, (v1, v2) in enumerate(zip(genes, pearsons)):
        print(v1, v2)
        if idx > max_item:
            break

check_best_n("coad", cmmn_best.values, best_pearsons[0].values)
check_best_n("hvg", hvg_best.values, best_pearsons[1].values)
check_best_n("icms2up", icms2up_best.values, best_pearsons[2].values)
check_best_n("icms2down", icms2down_best.values, best_pearsons[3].values)
check_best_n("icms3up", icms3up_best.values, best_pearsons[4].values)
check_best_n("icms3down", icms3down_best.values, best_pearsons[5].values)

