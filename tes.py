t = ['goose', 'duck', 'duck', 'goose']

def most_common(lst):
    return max(set(lst), key=lst.count)

print(most_common(t))