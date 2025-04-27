def IsKey(aListofValues):
    seen = set()
    return not any(i in seen or seen.add(i) for i in aListofValues)

print(IsKey("ABCDEF"))
print(IsKey("ABACDEF"))



def Fac():
    a = int(input("sayı girin:"))
    deger=1

    if a ==0 :
        return 1
    else:
        for i in range(a):
            deger = deger * (i+1)
        print(deger)

Fac()
