import pandas as pd
import os

path = r"c:\Users\owusu\Skripting\Bilder\eBKPH_Gesamttabelle.csv"
print('Exists:', os.path.exists(path))
df = pd.read_csv(path, sep=';', encoding='utf-8')
print('Rows:', len(df))
print(df.head())

# build tree

def insert_into_tree(tree: dict, code: str, bezeichnung: str):
    parts = [seg.strip() for seg in code.split('.') if seg.strip()]
    node = tree; acc = []
    for i, seg in enumerate(parts):
        acc.append(seg); key = '.'.join(acc)
        if 'children' not in node:
            node['children'] = {}
        if key not in node['children']:
            node['children'][key] = {'code': key, 'bez': (seg if i < len(parts)-1 else bezeichnung), 'children': {}}
        node = node['children'][key]
        if i == len(parts)-1:
            node['bez'] = bezeichnung


def build_tree_from_df(df):
    tree = {'code':'root', 'bez':'root', 'children': {}}
    for _, r in df.iterrows():
        insert_into_tree(tree, r['Code'], r['Bezeichnung'])
    return tree

tr = build_tree_from_df(df)
print('Top children count:', len(tr.get('children', {})))
print('Sample child keys:', list(tr.get('children', {}).keys())[:20])

# Check a nested key
sample = 'A01.01'
found = sample in tr.get('children', {})
print('Sample key A01.01 directly under root?', found)
# Find nested

def find_node(tree, key):
    if tree.get('code') == key:
        return tree
    for child in tree.get('children', {}).values():
        res = find_node(child, key)
        if res:
            return res
    return None

node = find_node(tr, sample)
print('Found node A01.01 via recursive search?', node is not None)
if node:
    print(node)
