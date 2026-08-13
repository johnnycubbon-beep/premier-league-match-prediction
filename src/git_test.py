import copy 
L = [[1,2], [3,2], [9,8,7]]
L1 = copy.copy(L)
L2 = copy.deepcopy(L)
L3 = L
L3[1]=5


print(L==L1)
print(L==L2)
print(L==L3)



def remove_elem(L,e):
    while e in L:
            L.remove(e)

La = [1,1,1,1,2,2,3]
remove_elem(L=La,e=3)
print(La)
