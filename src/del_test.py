L = [e for e in range(10) if e%3 == 0 or e%2 == 0]
for e in L:
    L.remove(e)
print(L)