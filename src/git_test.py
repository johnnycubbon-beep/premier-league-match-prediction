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

La = [1,1,1,1,2,2]
Lb = [2,3,4,2,1,3,2]
# remove_elem(L=La,e=3)
# print(La)

def merge_lists(L1: list =[0,0], L2: list = [0,0]):
    """
    merge_lists take in two lists and modifies L1 such that the nth 
    entry of L1 is a list of the original L1 entry and the corresponding L2 entry
    """
    if len(L1) != len(L2):
         raise Exception("Lists need to be of the same length!")
    for i in range(len(L1)):
         L1[i] = [L1[i], L2[i]]

merge_lists(L1=La, L2=Lb)
print(La)
         
         

