from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# CSDL giao dịch
transactions = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "D", "E"],
        ["E", "D"],
        ["E", "C"],
        ["A", "D", "E"]
    ]

# Áp dụng thuật toán apriori để tìm tập phổ biến
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_sets = apriori(df, min_support=0.3, use_colnames=True)
print("Frequent sets:")
print(frequent_sets)

# Tìm các luật kết hợp từ tập phổ biến
rules = association_rules(frequent_sets, metric="confidence", min_threshold=1)
print("Rules:")
print(rules[['antecedents', 'consequents', 'confidence']])