from co6co.data import fibonacci, take, primes_fast
def test_fibonacci(): 
    print(list(take(fibonacci(), 9)))
    assert list(take(fibonacci(), 10)) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
def test_primes_fast(): 
    print(list(take(primes_fast(), 10000)))
    assert list(take(primes_fast(), 10)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]