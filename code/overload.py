class A(object):
    def __init__(self, a, b=None):
        if b == None:
            self.data = a
        else:
            self.data = a+b
if __name__ == '__main__':
    a = A(1)
    print a.data
    b = A(1, 2)
    print b.data 
