import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,association_rules

# Đọc dữ liệu từ CSDL, ví dụ ở đây là file csv
transactions = [
        ["A", "B", "D", "E"],
        ["B", "C", "E"],
        ["A", "B", "D", "E", "F"],
        ["A", "B", "C", "E"],
        ["B", "C", "D", "F"],
        ["A", "B", "C", "D", "E"]
    ]


te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_sets = fpgrowth(df, min_support=0.5, use_colnames=True)

print("Frequent sets:")
print(frequent_sets)

min_confidence = 0.8
rules = association_rules(frequent_sets, metric='confidence', min_threshold=min_confidence)
print("Rules:")
print(rules[['antecedents', 'consequents', 'confidence']])