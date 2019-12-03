"""Hi! Welcome to my math library. Why would you use this?
There's a perfectly good built in one? 
Oh well, thanks for coming, good luck deciphering what 
I've coded: I am a fan of dense code. Also, I'm bad at this
so my code is all over the place. 

But I am pretty proud of my Matrix and Polynom classes,
so thats something. Enjoy your stay. 

Peace. 

Miah - jmiah120@gmail.com
"""
###############################################################
## To-do : [ ] Make multinomial class
##         [ ] Finish multivar numint
##         [ ] Make multivar derivative?
##         [X] Make lambda integral (definite)
##         [ ] Polynom roots      
##         [ ] eigen nonsense      
##         [X] column space        
##         [X] null space          
##         [ ] Vec span          
##         [ ] Polynom truediv
##         [ ] Rational class
##         [X] Gamma func (yeah it counts as math, shut up) 
##         [X] Error function
##         [X] Beta function
##         [ ] Evaluation w work
##              ( ) In written work
##              ( ) In tex format
##         [ ] Etc. 
##

####################################
##                                ##
##                                ##
#      "Some Important Numbers"       
##                                ##
##                                ##
####################################

e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353547594571382178525166427427466391932003059921817413596629043572900334295260595630738132328627943490763233829880753195251019011573834187930702154089149934884167509244761460668082264800168477411853742345442437107539077744992069551702761838606261331384583000752044933826560297606737113200709328709127443747047230696977209310141692836819025515108657463772111252389784425056953696
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703
ln2 = 0.69314718055994530941723212145817656807550013436025
LN15 = 0.4054651081081644

#####################################
##                                 ##
##                                 ##
#         "Numeric functions"       
##                                 ##
##                                 ##
#####################################

def ciel(x,places=0):
    """Returns ceiling of a real number x.
    If x is an iterable, it wil return the
    ceiling of each of its elements."""
    k = 10**places
    if type(x) in (float,int): return int(k*x)//k if int(x) == x else int(k*x)//k+1
    elif type(x) in (iter,tuple,set,list): return list(ciel(i) for i in x)
    else: raise TypeError('Bro, idk what to do with that')

def floor(x,places=0):
    """Returns floor of a real number x.
    If x is an iterable, it wil return the
    floor of each of its elements."""
    k = 10**places
    if type(x) in (float,int): return int(k*x)//k
    elif type(x) in (iter,tuple,set,list): return list(floor(i) for i in x)
    else: raise TypeError('Bro, idk what to do with that')

def fact(n):
    "Returns n! for non-negative integer n"
    if type(n) is not int: raise TypeError('Argument must be an int. Use gamma for non-ints')
    return ifint(prod(i+1 for i in range(n)))

def perm(n,r):
    """Returns nPr for non-negative integers
    n and r, where n>=r"""
    if (type(n) is not int) or (type(r) is not int): raise TypeError('Both arguments must be ints')
    return ifint(prod(i+1 for i in range(n-r,n)))

def choose(n,r):
    """Returns nCr for non-negative integers
    n and r, where n>=r"""
    if (type(n) is not int) or (type(r) is not int): raise TypeError('Both arguments must be ints')
    return ifint(perm(n,r)//fact(r))

def lcm(n,m):
    """Returns the lowest common multiple of
    non-negative integers n and m"""
    if (type(n) is not int) or (type(m) is not int): raise TypeError('Both arguments must be ints')
    return min([m*(j+1) for j in range(n) if m*(j+1) in [n*(j+1) for j in range(m)]])

def gcd(n,m):
    """Returns the greates common factor of
    integers n and m"""
    if (type(n) is not int) or (type(m) is not int): raise TypeError('Both arguments must be ints')
    return max(n,m) if 0 in {n,m,n-m} else gcd(min(abs(n),abs(m)),max(abs(n),abs(m))%min(abs(n),abs(m)))

def isPrime(n):
    "Returns True if n is prime, else False"
    if type(n) is not int: raise TypeError('Argument must be int')
    return pfactor(n)==[n] 

def isCoprime(m,n):
    """Returns True if n is coprime with m,
    else False"""
    if (type(n) is not int) or (type(r) is not int): raise TypeError('Both arguments must be ints')
    return gcd(n,m) == 1 

def ifint(x):
    "Returns int form of x if no decimals, else just x"
    return int(x) if int(x)==x else x

def isint(x):
    "Returns True if x has no decimals, else False"
    return int(x)==x 

def prod(seq):
    "Returns the product of the elements of seq"
    p = 1
    if type(seq) in (int,float): return seq
    for i in seq:
        if type(i) not in (int,float,iter,list,set,tuple): raise TypeError('Elements must be numbers, dummy')
        p *= prod(i)
    return p

def sqrt(x):
    "Returns the square root of x"
    if type(x) not in (int,float): raise TypeError('Argument must be a number')
    return x**(0.5)

def pfactor(n):
    """Returns prime factorization as a list
    with repeats such that prod(pfactor(n))=n
    ... theoretically"""
    if type(n) is not int: raise TypeError('Argument must be an integer')
    p = [] if n == 1 else [n]
    for i in range(2,ciel(sqrt(n))+1):
        if n%i == 0: return [i]+pfactor(int(n/i))
    return [1] if n==1 else p

def factors(n):
    """Returns all factors of an integer n
    as a sorted list"""
    if type(n) is not int: raise TypeError('Argument must be an integer')
    return sorted({i for i in range(1,ciel(sqrt(n))+1) if n%i == 0}.union({n//i for i in range(1,ciel(sqrt(n))+1) if n%i == 0}))

def isPerfect(n):
    """Returns true if an integer n is
    perfect, else false"""
    return sum(factors(n))==2*n   

def isAbundant(n):
    """Returns true if an integer n is
    abundant, else false"""
    return sum(factors(n))>2*n    

def isDeficient(n):
    """Returns true if an integer n is
    deficient, else false"""
    return sum(factors(n))<2*n    

def isPerf(n):
    """Returns 'Perfect' if n is perfect,
    'Abundant' if n is abundant, or
    'Deficient' if n is deficient"""
    return "Abundant" if isAbundant(n) else "Perfect" if isPerfect(n) else "Deficient"

def isPower2(x):
    """Returns true if x=2^n for some int n
    else false"""
    return isint(log(abs(x), base=2))

def isPowern(x,n):
    """Returns true if x=n^m for some int m
    else false"""
    return isint(log(abs(x), base=n))

def binomexp(a,b,n):
    """Returns the expansion of (a+b)^n as a list.
    a and b can be str,int,or float. Let a=b=1 for
    list of coeffs"""
    if (type(a)==str)and(type(b)==str): return list(f"{choose(n,i)}({a}^{n-i})({b}^{i})" for i in range(n+1))
    elif type(a)==str: return list(f"{choose(n,i)*(b**(i))}({a}^{n-i})" for i in range(n+1))
    elif type(b)==str: return list(f"{choose(n,i)*(a**(n-i))}({b}^{i})" for i in range(n+1))
    else: return list(ifint(choose(n,i)*(a**(n-i))*(b**i)) for i in range(n+1))
    

####################################
##                                ##
##                                ##
#      "Exponential Nonsense"       
##                                ##
##                                ##
####################################

def scinot(x):
    """Returns x as a tuple (A,n)
    where x=A*10^n"""
    A,n = x,0
    while A >= 10: A,n = A/10,n+1
    while A < 1: A,n = A*10,n-1
    return (A,n)

def binscinot(x):
    """Returns x as a tuple (A,n)
    where x=A*2^n"""
    n = 0
    while x >= 2**10: x,n = x/2**10,n+10
    while x >= 2: x,n = x/2,n+1
    while x < 1: x,n = x*2,n-1
    return (x,n)
    
def lnnot(x):
    "internal nonsense"
    n = 0
    while x >= 1.5**10: x,n = x/1.5**10,n+10
    while x >= 1.5: x,n = x/1.5,n+1
    while x < 1: x,n = x*1.5,n-1
    return (x,n)

def ln(x):
    """Returns natural log of x
    accurate to about 14 decimal places"""
    if x<0: raise ValueError('No real log for x<0')
    if x>1.5: 
        A,n = lnnot(x)
        return n*LN15+ln(A)
    elif x>1:
        a = x+2*((x-(e**x))/(x+(e**x)))
        a += 2*((x-(e**a))/(x+(e**a)))
        return a+2*((x-(e**a))/(x+(e**a)))
    elif x in {1,0,2}: return {1:0,0:'-inf',2:ln2}[x]
    else: return -ln(1/x)

def log(x,base=10):
    "Returns log base [base] of x"
    return ln(x)/ln(base)

##arcsin, arccos, arctan,
##arccsc, arcsec, arccot,
##archsin, archcos, archtan,
##archcsc, archsec, archcot

#####################################
##                                 ##
##                                 ##
#     "Trigonometric functions"       
##                                 ##
##                                 ##
#####################################

def sin(x):
    "Returns sin(x) for a real number x in radians"
    x = x%(2*pi)
    return sum(((-1)**i)*(x**(2*i+1))/fact(2*i+1) for i in range(0,20))
def cos(x):
    "Returns cos(x) for a real number x in radians"
    x = x%(2*pi)
    return sum((-1)**i*x**(2*i)/fact(2*i) for i in range(0,20))
def tan(x):
    "Returns tan(x) for a real number x in radians"
    x = x%pi
    return inf() if cos(x) == 0 else sin(x)/cos(x)
def csc(x):
    "Returns csc(x) for a real number x in radians"
    return inf() if sin(x) == 0 else 1/sin(x)
def sec(x):
    "Returns sec(x) for a real number x in radians"
    return inf() if cos(x) == 0 else 1/cos(x)
def cot(x):
    "Returns cot(x) for a real number x in radians"
    return inf() if sin(x) == 0 else cos(x)/sin(x)
def sinh(x):
    "Returns sinh(x) for a real number x"
    return 0.5(e**x-e**(-x))
def cosh(x):
    "Returns cosh(x) for a real number x"
    return 0.5(e**x+e**(-x))
def tanh(x):
    "Returns tanh(x) for a real number x"
    return inf() if cosh(x) == 0 else sinh(x)/cosh(x)
def csch(x):
    "Returns csch(x) for a real number x"
    return inf() if sinh(x) == 0 else 1/sinh(x)
def sech(x):
    "Returns sech(x) for a real number x"
    return inf() if cosh(x) == 0 else 1/cosh(x)
def coth(x):
    "Returns coth(x) for a real number x"
    return inf() if sinh(x) == 0 else cosh(x)/sinh(x)
def arcsin(x):
    "Returns arcsin(x) in radians for a real number x Doesn't work rn "
    return -arcsin(-x) if x<0 else pi/2 if x == 1 else sum(
        (x**(2*n+1))*prod(list(
            (2*i-1)**2 for i in range(2,n))
                          )/fact(2*n+1) for n in range(85))

####################################
##                                ##
##                                ##
# "Some other important functions"       
##                                ##
##                                ##
####################################

def numint(func,a,b,n=1000,rule='simp'):
    """Returns the integral of func from a to b
    with n rects/trapz/parabs."""
    delta = (b-a)/n
    if   rule == 'simp' : return delta/6*(2*sum(func(a+delta*x/2) for x in range(1,2*n,2))+4*sum(func(a+delta*x/2) for x in range(2,2*n,2))+sum((func(a),func(b))))
    elif rule == 'zoid' : return delta*(sum(func(a+delta*x) for x in range(1,n))+sum((func(a),func(b)))/2)
    elif rule == 'mid'  : return delta*sum(func(a+delta*(x+1/2)) for x in range(n))             
    elif rule == 'left' : return delta*sum(func(a+delta*x) for x in range(n))
    elif rule == 'right': return delta*sum(func(a+delta*(x+1)) for x in range(n))
    
def gamma(z):
    """Returns Gamma function of x.
    Accurate to about 13 decimal places."""
    qs=[75122.6331530,80916.6278952,36308.2951477,8687.24529705,1168.92649479,83.8676043424,2.50662827511]
    a = sum(q*(z**n) for n,q in enumerate(qs))
    b = prod(z+n for n,q in enumerate(qs))
    c = ((z+5.5)**(z+0.5))*e**(-z-5.5)
    if isint(z): return fact(abs(z-1))
    elif z>1.5: return (z-1)*gamma(z-1)
    else:
        return (a/b)*c/((1-5.234230557400197e-13)*(1+6.863196879500968e-15))
        
def beta(a,b):
    "Returns Beta function of x"
    return gamma(a)*gamma(b)/gamma(a+b)

def erf(x):
    """Returns the error function at x.
    Accurate to the given decimal"""
    if x>=0:
        p,a1,a2,a3,a4,a5 = 0.3275911,0.254829592,-.284496736,1.421413741,-1.453152027,1.061405429
        t = 1/(1+p*x)
        return round(1-(a1*t+a2*t**2+a3*t**3+a4*t**4+a5*t**5)*e**(-(x**2)),6)
    else: return -erf(-x)

####################################
##                                ##
##                                ##
#        "Polynomial Class"       
##                                ##
##                                ##
####################################

class Polynom:
    """Creates a polynomial object with args
    as the coefficients and term degree in
    ascending order  (little endian) """
    def __init__(self,*args):
        self.args = args if args != () else (0,) 
        self.func = lambda x: sum(j*(x**i) for i,j in enumerate(args))
        self.degree = len(args)-1
    def __add__(self, polynom):
        "Returns self+polynom"
        if type(polynom) == Polynom:
            arg1,arg2 = len(self.args),len(polynom.args)
            if arg1 != arg2:
                self.args += (0,)*(max(arg1,arg2)-min(arg1,arg2))
                polynom.args += (0,)*(max(arg1,arg2)-min(arg1,arg2))
            args = (i+j for i,j in zip(self.args,polynom.args))
        elif type(polynom) in {int, float}:
            args = (self.args[0]+polynom,)+self.args[1:]
        return Polynom(*args)
    def __sub__(self, polynom):
        "Returns self-polynom"
        if type(polynom) == Polynom:
            arg1,arg2 = len(self.args),len(polynom.args)
            if arg1 != arg2:
                self.args += (0,)*(max(arg1,arg2)-min(arg1,arg2))
                polynom.args += (0,)*(max(arg1,arg2)-min(arg1,arg2))
            args = (i-j for i,j in zip(self.args,polynom.args))
        elif type(polynom) in {int, float}:
            args = (self.args[0]-polynom,)+self.args[1:]
        return Polynom(*args)
    def __mul__(self,polynom):
        "Returns self*polynom"
        if type(polynom) == Polynom:
            new_arg = tuple()
            for i,j in enumerate(self.args):
                #the 0 tuple ensures proper degree, the j*k ensures proper coefficients
                arg = tuple(0 for k in range(i))+tuple(j*k for k in polynom.args)
                new_arg = Polynom._tup_add(arg,new_arg)
        elif type(polynom) in {int, float}:
            new_arg = tuple(polynom*i for i in self.args)
        return Polynom(*new_arg)
    def __pow__(self, n):
        "Returns self**n for an integer n, returns 1 for n<=0"
        return prod(list(self for i in range(n))) if n > 0 else Polynom(1) 
    def __floordiv__(self,polynom):
        "Returns self//polynom"
        if type(polynom) == Polynom:
            arg1 = Polynom._tup_strip(self.args)
            arg2 = Polynom._tup_strip(polynom.args)
            lst = [0]*(self.degree-polynom.degree+1)
            i = -1
            while len(arg1) >= len(arg2):
                i += 1
                c = arg1[-1]/arg2[-1]
                lst[i] = c
                arg1 = Polynom._tup_add(arg1,tuple(0 for i in range(len(arg1)-len(arg2)))+tuple(-c*i for i in arg2))
                arg1 = Polynom._tup_strip(arg1)
            lst.reverse()
            tup = tuple(lst)
        elif type(polynom) in {int, float}:
            tup = tuple(i/polynom for i in self.args)
        return Polynom(*tup)
    def __truediv__(self,polynom):
        "Returns self/n for a number n, polynom functionality coming soon"
        ########################
        ## do this eventually ##
        ########################
        if type(polynom) == Polynom:
            pass
        elif type(polynom) in {int, float}:
            tup = tuple(i/polynom for i in self.args)
            return Polynom(*tup)
    def __mod__(self,polynom):
        "Returns self%polynom"
        p = self-((self//polynom)*polynom)
        return Polynom(*(Polynom._tup_strip(p.args)))
    def __str__(self):
        "Returns self as we'd write it"
        s = ""
        for i,j in enumerate(self.args):
            if i == 0 & j!=0: s += f"{j} + "
            elif i == 1: s += "x + " if j==1 else f"{j}x + " if j!=0 else ''
            else: s += f"x^{i} + " if j==1 else f"{j}x^{i} + " if j!=0 else '' 
        return s[:-3]
    def __call__(self,arg):
        "Returns self(arg) => f(x)"
        return self.func(arg)
    def __repr__(self):
        "Returns self as command"
        return "Polynom{}".format(self.args)
    def __eq__(self,polynom):
        "Returns True if self=polynom else False"
        return True if self.args == polynom.args else False 
    def __lt__(self,polynom):
        "Returns True if self<polynom else False"
        return True if self.degree < polynom.degree else False 
    def __le__(self,polynom):
        "Returns True if self<=polynom else False"
        return True if self.degree <= polynom.degree else False 
    def __ne__(self,polynom):
        "Returns True if self!=polynom else False"
        return True if self.degree != polynom.degree else False 
    def __ge__(self,polynom):
        "Returns True if self>=polynom else False"
        return True if self.degree > polynom.degree else False 
    def __gt__(self,polynom):
        "Returns True if self>polynom else False"
        return True if self.degree >= polynom.degree else False
    def __getitem__(self,item):
        c, e = self.args[item], item 
        if c == 0: return "0"
        elif c == 1 and e != 1: return f"x^{e}"
        elif e == 1: return f"{c}x"
        elif e == 0: return f"{c}"
        else: return f"{c}x^{e}"
    def __or__(self,polynom):
        ## use to find intersection of two polynoms
        pass
    def integral(self,a,b):
        """Returns the integral from a to b of self, if a or b
        are none, returns lambda function of indefinite integral"""
        intfunc = lambda x: sum(j*(x**(i+1))/(i+1) for i,j in enumerate(self.args))
        if (a,b)==(None,None): new_args = (0,)+tuple(j/(i+1) for i,j in enumerate(self.args))
        elif a==None: new_args = (intfunc(b),)+tuple(-j/(i+1) for i,j in enumerate(self.args))
        elif b==None: new_args = (intfunc(a),)+tuple(j/(i+1) for i,j in enumerate(self.args))
        else: return intfunc(b)-intfunc(a)
        return Polynom(*new_args)
    def ddx(self):
        "Returns df/dx"
        new_args = tuple(j*i for i,j in enumerate(self.args))[1:]
        return Polynom(*new_args)
    @classmethod
    def zero(cls):
        "Returns the 0 polynom"
        return cls(0)
    @staticmethod
    def _tup_add(tup1, tup2):
        "Adds two tuples the vector way"
        arg1 = len(tup1)
        arg2 = len(tup2)
        if arg1 != arg2:
            tup1 += (0,)*(max(arg1,arg2)-min(arg1,arg2))
            tup2 += (0,)*(max(arg1,arg2)-min(arg1,arg2))
        args = tuple(i+j for i,j in zip(tup1,tup2))
        return args
    def _tup_strip(tup):
        "Removes trailing 0's in a tuple"
        try:
            while tup[-1]==0:
                tup = tup[:-1]
        except: pass
        finally: return tup

####################################
##                                ##
##                                ##
#  "Multivariable Equation Class"       
##                                ##
##                                ##
####################################

class Multivar:
    def __init__(self,expr,*params):
        s = ''
        for i in expr:
            s += '**' if i=='^' else i
        exec("self.func = lambda {}: {}".format(",".join(params),s))
        self.s = expr
        self.params = params
    def __repr__(self):
        return self.s
    def __str__(self):
        return self.s
    def numint(self,bounds,n=1000,rule='mid'):
        """Returns the integral of func from the bounds
        with n rects per dimension. bounds should be a
        dict with key->param, val->(a,b)"""
##        for i in bounds:
##            exec(f"delta_{i} = {(bounds[i][0]-bounds[i][0])/n}")
##            for x in range(*bounds[i]):
##                pass
##        elif rule == 'mid'  : return delta*sum(func(a+delta*(x+1/2)) for x in range(n))             
##        elif rule == 'left' : return delta*sum(func(a+delta*x) for x in range(n))
##        elif rule == 'right': return delta*sum(func(a+delta*(x+1)) for x in range(n))
    
###################################
##                               ##
##                               ##
#    "Rational Equation Class"       
##                               ##
##                               ##
###################################

class Rational:
    def __init__(num,denom):
        pass
    def __add__(self, polynom):
        pass
    def __sub__(self, polynom):
        pass
    def __mul__(self,polynom):
        pass
    def __pow__(self, n):
        pass
    def __floordiv__(self,polynom):
        pass
    def __truediv__(self,polynom):
        pass
    def __mod__(self,polynom):
        pass
    def __str__(self):
        pass
    def __call__(self,arg):
        pass
    def __repr__(self):
        pass
    def __eq__(self,polynom):
        pass
    def __lt__(self,polynom):
        pass
    def __le__(self,polynom):
        pass
    def __ne__(self,polynom):
        pass
    def __ge__(self,polynom):
        pass
    def __gt__(self,polynom):
        pass

####################################
##                                ##
##                                ##
#      "Matrix Handler Class"       
##                                ##
##                                ##
####################################

class Matrix:
    def __init__(self,rows,columns,*array):
        self.array = list(array)
        self.ro = rows
        self.co = columns
        if len(array) < self.ro*self.co:
            array+(0,)*(self.ro*self.co-len(array))
        array = list(array)
        self.mat = []
        for i in range(self.ro):
            self.mat.append(list([array.pop(0) for i in range(self.co)]))
    def __str__(self):
        return "".join(f"{list(ifint(round(x,2)) for x in i)}\n" for i in self)
    def __repr__(self):
        return "".join(f"{i}\n" for i in self)
    def __add__(self, other):
        if (self.co,self.ro)!=(other.co,other.ro):
            raise ArithmeticError("Matrices must have the same size to add them")
        new = tuple(self.array[i]+other.array[i] for i in range(self.m*self.n)) 
        return Matrix(self.ro,self.co,*new)    
    def __sub__(self,other):
        if (self.co,self.ro)!=(other.co,other.ro):
            raise ArithmeticError("Matrices must have the same size to subtract them")
        new = tuple(self.array[i]-other.array[i] for i in range(self.m*self.n)) 
        return Matrix(self.ro,self.co,*new)    
    def __mul__(self,other):
        if type(other) in {float,int}:
            args = ()
            for i in self:
                args += tuple(other*x for x in i)
            return Matrix(self.ro,self.co,*args)
        if self.co!=other.ro:
            raise ArithmeticError("Matrices not compatible for multiplication")
        new = []
        for i,j in [(i,j) for i in range(self.ro) for j in range(other.co)]:
            new.append(sum(self[i][k]*other[k][j] for k in range(self.co)))
        return Matrix(self.ro,other.co,*new)
    def __pow__(self, n):
        return prod(self for i in range(n))
    def __eq__(self,other):
        return True if self.mat==other.mat else False
    def __ne__(self,other):
        return True if self.mat!=other.mat else False
    def __getitem__(self, item=None):
        "Use [None][:COLUMN:] to access a column"
        if item==None: return self.trans()
        else: return self.mat[item]
    def __setitem__(self,entry,item):
        self.mat[entry] = item
    def __mod__(self,value):
        if type(value) in {float,int}:
            args = tuple(i%value for i in self.array)
            return Matrix(self.ro,self.co,*args)
    def ind(self,i,j):
        return self[i][j]
    def copy(self):
        return Matrix(self.ro,self.co,*self.array)
    def det(self):
        "Returns the determinant of self"
        if self.co == self.ro:
            if self.co == 2:
                a,b,c,d = self.array
                return a*d-b*c
            else:
                return sum(((-1)**i)*self.ind(0,i)*self.cofactor(0,i).det() for i in range(self.co))
    def cofactor(self,i,j):
        mat = list(i.copy() for i in self)
        mat.pop(i)
        for k in mat:
            k.pop(j)
        new_args = ()
        for col in mat:
            new_args += tuple(col)
        return Matrix(self.ro-1,self.co-1,*new_args)
    def concat(self,other):
        "Concatenates other to self. Really only used for inverses"
        concatmat = list(i+j for i,j in zip(self,other))
        args = ()
        for i in concatmat:
            args += tuple(i)
        return Matrix(self.ro,self.co+other.co,*args)
    def colelim(self,column):
        for i in self:
            i.pop(column)
        for j in range(self.ro):
            self.array.pop(column+j*self.co)
        self.co -= 1
        return self
    def inv(self):
        "Returns inverse matrix"
        if self.ro != self.co or self.det()==0:
            return None # :'(
        interim = self.concat(Matrix.I(self.ro)).rref()
        for i in range(self.ro):
            interim.colelim(0)
        return interim
    def trans(self):
        "Returns transpose matrix"
        new_args = ()
        for i in range(self.co):
            new_args += tuple(self[j][i] for j in range(self.ro))
        return Matrix(self.co,self.ro,*new_args)
    def scale(self,scalar,linescale,linewrite):
        """Multiplies linescale of Matrix by scalar
        writes to linewrite *IN PLACE*"""
        linescale = self.mat[linescale].copy()
        self[linewrite] = list(ifint(scalar*i) for i in linescale)
        a,b = linewrite*self.co, linewrite*self.co+self.co
        self.array[a:b] = list(ifint(scalar*i) for i in linescale)
        return self
    def elim(self,linewrite,linelim,scalar=1):
        """Subtracts scalar*linelim from linewrite
        writes to linewrite *IN PLACE*"""
        linelim = self.mat[linelim].copy()
        self[linewrite] = list(self[linewrite][i]-scalar*j for i,j in enumerate(linelim))
        a,b = linewrite*self.co, linewrite*self.co+self.co
        self.array[a:b] = list(self[linewrite][i]-scalar*j for i,j in enumerate(linelim))
        return self
    def ref(self):
        "Returns matrix in Roe-Echelon form"
        mat = self.mat
        lim = min(self.ro,self.co)
        for i in range(lim):
            try: self.scale((1/mat[i][i]),i,i)
            except ZeroDivisionError: self.scale(1,i,i)
            for j in range(i+1,lim):
                self.elim(j,i,mat[j][i])
        for i in range(self.ro-self.co):
            for j in range(self.co):
                self.elim(self.co+i,j,self[self.co+i][j])
            self.scale(1,self.co+i,self.co+i)
        return self
    def rref(self):
        "Returns matrix in reduced Roe-Echelon form"
        self.ref()
        lim = min(self.ro,self.co)
        for i in range(self.ro):
            for j in range(i+1,lim):
                self.elim(i,j,self[i][j])
            self.scale(1,i,i)
        return self
    def colspace(self):
        "Returns dimension of column space"
        lim = min(self.co,self.ro)
        interim = self.copy().rref()
        return list(interim[i][i] for i in range(lim)).count(1)
    def nulspace(self):
        "Returns dimension of null space"
        return self.ro-self.colspace()
    def eigspace(self):
        #idek babe
        pass
    @classmethod
    def zero(cls,n,m):
        return cls(n,m,*(0 for i in range(n*m)))
    @classmethod
    def I(cls,n):
        array = ()
        for i,j in [(i,j) for i in range(n) for j in range(n)]:
            array += (0,) if i!=j else (1,)
        return cls(n,n,*array)

####################################
##                                ##
##                                ##
#       "Vector Handler Class"       
##                                ##
##                                ##
####################################

class Vec:
    def __init__(self,*args):
        self.args = args
    def __add__(self,other):
        "Returns self+other"
        return Vec(*tuple(i+j for i,j in zip(self.args,other.args)))
    def __sub__(self,other):
        "Returns self-other"
        return Vec(*tuple(i-j for i,j in zip(self.args,other.args)))
    def __mult__(self,other):
        "Returns Haddamard product of self and other"
        return Vec(*tuple(i*j for i,j in zip(self.args,other.args)))
    def cross(self,other):
        """Returns the cross product of self and other
        only works in 3 dimensions"""
        x, y, z = self.args
        a, b, c = other.args
        return Vec((y*c-b*z),(x*c-a*z),(x*b-a*y))
    def dot(self,other):
        "Returns the dot product of self and other"
        return sum((self*other).args)
    def mag(self):
        "Returns the magnitude of the vec"
        return sqrt(sum(i**2 for i in self.args))
    def __len__(self):
        "Returns the dimension of the vector"
        return len(self.args)
    @classmethod
    def span(*vecs):
        dim = max(len(i) for i in vecs)
        for i in vecs:
            if len(i)<dim: i = Vec(*(i.args+(0,)*(dim-len(i))))      
        return 0

####################################
##                                ##
##                                ##
#            "infinity"            
##                                ##
##                                ##
####################################

class inf:
    "Object to represent dummy thicc numbers"
    def __str__(self):
        return 'inf'
    def __repr__(self):
        return 'inf'
    def __sub__(self):
        return '-inf'

def test(func,mathfunc,inp,inp2=None):
    from time import perf_counter as clock
    exec(f"from math import {mathfunc} as mfunc") 
    from random import random as r
    t0 = clock()
    for i in range(1,500):
        exec(f"i = {inp}")
        mfunc(i)
    t1 = clock()-t0
    for i in range(1,500):
        exec(f"i = {inp}")
        func(i)
    t2 = clock()-t0-t1
    print(f"math func := {t1}")
    print(f"my   func := {t2}")
    print(f"builtin {t2/t1} times as efficient as me")
    for i in range(1,20):
        if inp2==None: exec(f"i = {inp}")
        else: exec(f"i = {inp2}")
        print(f"Error = {1-(func(i)/mfunc(i))}")



    
