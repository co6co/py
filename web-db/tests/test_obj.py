def test_dict():
    d={}
    if d:
        print("d is not None")
    else:
        print("d is None")

    if d==None:
        print("d is None")

    else:
        print("d is not None")

    if d is not None:
        print("d is not None")
    else:
        print("d is None")
