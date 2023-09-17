import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
from operator import mul
from random import randint 

def full():
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

plt.fullscreen = full

################CLASSE POINT###################

class Point :
    """
    Permet de définir une instance comme un point dans un plan.
    """
    def __init__(self, x, y=None) :
        """
        Il est possible d'initialiser une instance à l'aide de
        deux coordonnées x et y, ou bien avec une affixe complexe.
        Exemples :
            - Point(4,5) -> (4;5)
            - Point(Complex(4,5)) -> (4;5)
        """
        if isinstance(x,complex) :
            self.x=x.real
            self.y=x.imag
        else :
            self.x=x
            self.y=y
    
    def __repr__(self):
        return f"({self.x};{self.y})"
    
    def affixe(self):
        return complex(self.x,self.y)
    
    def __add__(self, other) :
        assert type(other) in (Point, complex)
        if isinstance(other, Point) :
            return Point(self.affixe()+other.affixe())
        return Point(self.affixe()+other)
    
    def __sub__(self, other) :
        assert type(other) in (Point, complex)
        if isinstance(other, Point) :
            return Point(self.affixe()-other.affixe())
        return Point(self.affixe()-other)
    
    def __rsub__(self, other) :
        assert type(other) in (Point, complex)
        if isinstance(other, Point) :
            return Point(other.affixe()-self.affixe())
        return Point(other-self.affixe())
    
    def __neg__(self, other) :
        return Point(-self.affixe())

exposant = {0:"⁰",
            1:"¹",
            2:"²",
            3:"³",
            4:"⁴",
            5:"⁵",
            6:"⁶",
            7:"⁷",
            8:"⁸",
            9:"⁹"}

def puissance(numb) :
    return "".join([exposant[int(i)] for i in str(numb)])

def mindegree(a,b) :
    return min(len(a.coeff),len(b.coeff))
    
def maxdegree(a,b) :
    return max(len(a.coeff),len(b.coeff))


############################   Classe Fractions   ####################################



from math import gcd

class Fraction:
    """
    Classe imaginée pour abolir le calcul approximatif dans Python. Permet des opérations et simplifications de fraction.
    Possède une compatibilité optimisée avec les polynomes et les complexes.
    """
    def __init__(self, n, d=1):
        """Constructeur qui génère une erreur si le dénominateur est nul, ou si n et d ne sont pas des entiers"""
        try :
            assert d!=0
        except AssertionError :
            print("Wait... That's illegal !")
        else :
            if isinstance(n,Fraction) :
                n,d=n.numérateur, d*n.dénominateur
            if isinstance(d,Fraction) :
                n,d=n*d.dénominateur, d.numérateur
            self.numérateur=n
            self.dénominateur=d
            
    
    def __repr__(self):
        """Permet à Thonny et à Python de savoir comment représenter la fraction par exemple lors d'un print"""
        self.simplifier()
        return str(self.numérateur)+"/"+str(self.dénominateur)
    
    def __str__(self):
        """Méthode utilisée lors de l'affichage de polynômes à coefficient rationnels"""
        self.simplifier()
        return "("+str(self.numérateur)+"/"+str(self.dénominateur)+")"
    
    def __int__ (self) :
        return int(self.decimal())
    
    def __neg__(self) :
        return Fraction(-self.numérateur, self.dénominateur)
    
    def pgcd(self):
        """retourne un entier égal au PGCD du numérateur et du dénominateur"""
        return gcd(self.numérateur, self.dénominateur)
    
    def decimal(self) :
        return self.numérateur/self.dénominateur
    
    def __float__(self) :
        return self.decimal()
    
    def __round__(self, bord) :
        return round(self.decimal(), bord)
    
    def simplifier(self):
        """retourne la fraction égale simplifiée"""
        if self.numérateur/self.dénominateur==self.numérateur//self.dénominateur :
            return self.numérateur//self.dénominateur
        if self.dénominateur<0 : #Si on a du -n/-d ou du n/-d
            self.numérateur, self.dénominateur = -self.numérateur, -self.dénominateur #On remplace par n/d ou -n/d
        return Fraction(self.numérateur//self.pgcd(),self.dénominateur//self.pgcd())
    
    def __add__(self,other):
        """retourne la fraction égale à la somme de cette fraction avec other"""
        if isinstance(other, int) :
            other = Fraction(other)
        if isinstance(other, complex) :
            pass
        return Fraction(self.numérateur*other.dénominateur+self.dénominateur*other.numérateur, self.dénominateur*other.dénominateur).simplifier()
    
    __radd__ = __add__
    
    def __sub__(self,other) :
        return self+(-other)
    
    def __rsub__(self,other) :
        return other+(-self)
    
    def __mul__(self,other):
        """retourne la fraction égale au produit de cette fraction avec other"""
        if isinstance(other, float) :
            n=1
            while other*10**n!=int(other*10**n) :
                n+=1
            return Fraction(self.numérateur*int(other*10**n), self.dénominateur*10**n).simplifier()
        elif isinstance(other, Fraction) :
            return Fraction(self.numérateur*other.numérateur, self.dénominateur*other.dénominateur).simplifier()
        elif isinstance(other, Polynomial) :
            return other*self
        else :
            return Fraction(self.numérateur*other, self.dénominateur).simplifier()

    __rmul__ = __mul__
    
    def __pow__(self, n) :
        assert type(n)==int
        return Fraction(self.numérateur**n, self.dénominateur**n)
    
    def __truediv__(self, other):
        """retourne la fraction égale au produit de cette fraction avec other"""
        if isinstance(other, float) :
            return self.decimal()/other
        if isinstance(other, int) :
            other = Fraction(other)
        return Fraction(self.numérateur*other.dénominateur, self.dénominateur*other.numérateur).simplifier()

    def __rtruediv__ (self, other) :
        if isinstance(other, float) :
            return other/self.decimal()
        elif isinstance(other, int) :
            other = Fraction(other)
        return Fraction(self.dénominateur*other.numérateur, self.numérateur*other.dénominateur).simplifier()


########################### FRACTION-FRIENDLY COMPLEX CLASS ###########################

class Complex :
    """
    Une classe qui se veut identique à la classe complex buit-in du module cmath, mais qui possède une bonne compatibilité
    avec les fractions et les polynomes.
    Par exemple, il est possible de :
        - Calculer des images complexes par une fonction polynome
        - Créer des complexes dont les coefficients sont des fractions
        - Créer des polynomes dont les coefficients sont des complexes
    """
    def __init__(self, real, imag) :
        assert type(real) in (int, float, Fraction) and type(imag) in (int, float, Fraction)
        self.real = real
        self.imag = imag
    
    def __repr__(self):
        if int(self.real)==self.real :
            self.real=int(self.real)
        if int(self.imag)==self.imag :
            self.imag=int(self.imag)
        if self.imag==0 :
            self=self.real
            return str(self)
        return f" {self.real}+{self.imag}i".replace("-1i","-i").replace("+1i","+i").replace("+-","-").replace(" 0+","").replace(" 0-","-").replace(" ","")
    
    def __str__(self):
        if int(self.real)==self.real :
            self.real=int(self.real)
        if int(self.imag)==self.imag :
            self.imag=int(self.imag)
        if self.imag==0 :
            self=self.real
            return str(self)
        return f"( {self.real}+{self.imag}i)".replace("-1i","-i").replace("+1i","+i").replace("+-","-").replace(" 0+","").replace(" 0-","-").replace(" ","")
        
    def image(self) :
        return Point(self.real, self.imag)
    
    def __eq__(self, other) :
        return self.real== other.real and self.imag==other.imag
    
    def __add__(self, other) :
        if isinstance(other, int) or isinstance(other, float) or isinstance(other,Fraction):
            return Complex(self.real+other, self.imag)
        if isinstance(other, Polynomial) :
            return other+self
        else :
            return Complex(self.real+other.real, self.imag+other.imag)
    
    __radd__=__add__
    
    def __int__(self) :
        return int(self.real)
    
    def __round__(self, bord) :
        return Complex(round(self.real,bord), round(self.imag,bord))
    
    def __sub__(self,other) :
        return self+(-other)
    
    def __rsub__(self,other) :
        return other+(-self)
    
    def __neg__(self) :
        return Complex(-self.real, -self.imag)
    
    def conj(self):
        return Complex(self.real, -self.imag)

    def __mul__(self,other):
        if isinstance(other, int) :
            return Complex(self.real*other, self.imag*other)
        if isinstance(other, float) :
            return round(Complex(self.real*other, self.imag*other),5)
        if isinstance(other, Polynomial) :
            return other*self
        prod = Polynomial(self.imag, self.real)*Polynomial(other.imag, other.real)
        return prod(i)
    
    __rmul__=__mul__
    
    def __pow__ (self, n) :
        assert isinstance(n, int)
        if self==i :
            return [1,i,-1,-i][n%4]
        
        poly=Monome(0,1)
        for j in range(n) :
            poly*=Polynomial(self.imag, self.real)
        return poly(i)
    
    def __truediv__(self, other) :
        if isinstance(other, int) or int(other)==other :
            return Complex(Fraction(self.real, int(other)).simplifier(), Fraction(self.imag,int(other)).simplifier())
        elif isinstance(other, float) :
            return round(Complex(self.real/other, self.imag/other),5)
        if round(other,0)==other :
            return (self*other.conj())/round(abs(other)**2,0)
        return (self*other.conj())/(abs(other)**2)
    
    def __rtruediv__(self, other) :
        if isinstance(other, int) or int(other)==other :
            return Complex(Fraction(int(other),self.real).simplifier(), Fraction(int(other),self.imag).simplifier())
        elif isinstance(other, float) :
            return round(Complex(other/self.real, other/self.imag),5)
        if round(other,0)==other :
            return (other*self.conj())/round(abs(self)**2,0)
        return (other*self.conj())/(abs(self)**2)

    def __abs__(self) :
        n = (self.real**2+self.imag**2)**0.5
        if int(n)==n :
            n=int(n)
        return n

#########################   Classe Polynomial   #####################################


class Polynomial :
    """

    **************
    
    La classe Polynomial permett d'afficher et faire des opérations sur des polynomes de degré (positif ou nul) aussi grand que voulu.
    
    On initialise une instance de cette classe en rentrant soit la liste de ses coefficients, soit la *liste de ses coefficients (ses coefficients occupant chacun la place d'un argument), soit une chaine de charactères.
    
    - Pour initialiser une instance avec une liste ou *liste, tous les coefficients (mêmes nuls) doivent être rentrés par ordre décroissant du degré des monomes (comme on ordonnerait un polynome à l'écrit)
      Par exemple :
          Polynomial(4,8,5) --> 4x²+8x+5
          Polynomial(19,30,0,42) --> 19x³+30x²+42
    
    - Pour initialiser une instance avec une chaine de charactères, en notant n le coefficient d'un monome et d son degré, on doit entrer l'ensemble des monomes qui le composent, écrits sous la forme 'nxd' et séparés d'un signe '+'
      On doit rentrer tous les coefficients (même égals à 1, pas obligatoirement si nuls) et degrés (même nuls) pour mener l'initialisation à bien.
      Les monomes peuvent parfaitement être désordonnés, et plusieurs monomes de même degré peuvent apparaitre dans la chaine de charactères sans problème.
      Par exemple :
          Polynomial("3x5+2x4+5x2+4x0") --> 3x⁵+2x⁴+5x²+4
          Polynomial("5x2+7x6+14x1+12x2") --> 7x⁶+17x²+14x
    
    **************
    
    """
    def __init__(self, *coeff, var="x"):
        if isinstance(coeff[0],list):
            self.coeff = [coeff[0][n] for n in range(len(coeff[0])-1, -1,-1)]
        elif isinstance(coeff[0],str):
            self.coeff = sum([Monome(int(n.split("x")[1]), int(n.split("x")[0])) for n in coeff[0].split("+")]).coeff
        else :
            self.coeff = [coeff[n] for n in range(len(coeff)-1, -1,-1)]
        self.var = var
    
    def __repr__(self):
        return ((" "+"+".join([str(self.coeff[i])+"x"+puissance(i) for i in range(len(self.coeff)-1,0 ,-1) if self.coeff[i]!=0])).replace("+1x","+x").replace(" 1x", "x").replace(" ","")+"+"+str(self.coeff[0])+" ").replace("+-","-").replace("+0 "," ").replace("x¹+", "x+").replace("x¹-", "x-").replace("x¹ ", "x ").replace("+(0i)","").replace("+(-", "-(").replace("x",self.var)
    
    def __str__(self) :
        return ((" "+"+".join([str(self.coeff[i])+"x"+puissance(i) for i in range(len(self.coeff)-1,0 ,-1) if self.coeff[i]!=0])).replace("+1x","+x").replace(" 1x", "x").replace(" ","")+"+"+str(self.coeff[0])+" ").replace("+-","-").replace("+0 "," ").replace("x¹+", "x+").replace("x¹-", "x-").replace("x¹ ", "x ").replace("+(0i)","").replace("+(-", "-(").replace("x",self.var)
    
    def __call__(self, x, exact=True) :
        """

        **************
        
        Destiné à renvoyer l'image du réel x par la fonction polynome dont l'expression est celle de l'instance.
        Exemple :
            Avec f = Polynomial(4,8,5) --> 4x²+8x+5
            On a f(5) --> 145
            
            Remarque : on peut aussi écrire Polynomial(4,8,5)(5) 
            
        **************
        
        """
        if isinstance(x, str) :
            return Polynomial(self.coeff[::-1], var=x)
        if exact :
            return sum([self.coeff[n]*x**n for n in range(len(self.coeff))])
        return sum([self.coeff[n]*x**n if not(isinstance(self.coeff[n],Fraction)) else self.coeff[n].decimal()*x**n for n in range(len(self.coeff))])
        
    
    def __lt__(self,other):
        "Permet de comparer deux instances de la classe. Elles seront rangées dans le même ordre que leur degrés respectifs"
        return len(self.coeff)<len(other.coeff)
    
    def __add__(self, other):
        "Méthode d'addition entre deux instances de la classe, ou une instance et un réel"
        l=Polynomial(self.coeff[::-1])
        if not(isinstance(other,Polynomial)) :
            l.coeff[0]+=other
            return l
        reste = maxdegree(self,other)-mindegree(self,other)
        min(l,other).coeff+=(reste*[0])
        return Polynomial([l.coeff[n] + other.coeff[n] for n in range(len(l.coeff))][::-1])
    
    __radd__=__add__
    
    def __sub__(self,other):
        return self+(-other)
    
    def __rsub__(self, other) :
        return other+(-self)
    
    def __neg__(self):
        return Polynomial([-n for n in self.coeff][::-1])
    
    def __mul__(self, other):
        "Méthode de multiplication entre deux instances de la classe, ou une instance et un réel"
        if not(isinstance(other,Polynomial)) :
            return Polynomial([n*other for n in self.coeff][::-1])
        else :
            return sum([Monome(n+i, other.coeff[i]*self.coeff[n]) for i in range(len(other.coeff)) for n in range(len(self.coeff))])
    
    __rmul__=__mul__
    
    def __pow__ (self, exponent) :
        "Méthode d'élevation d'une instance de la classe à une puissance entière"
        assert isinstance(exponent, int)
        l = 1
        for i in range(exponent) :
            l*=self
        return l
    
    def __truediv__(self, other):
        "Méthode de division d'une instance de la classe par un réel"
        assert isinstance(other,int) or isinstance(other, float) or isinstance(other, Complex)
        return Polynomial([int(n/other) if int(n/other)==n/other else Fraction(n,other) if isinstance(other,int) else round(n/other,5) for n in self.coeff][::-1])

    def derivative(self, k=1) :
        if k==1 :
            return sum([Monome(n-1, n*self.coeff[n]) if len(self.coeff)!=1 else 0 for n in range(len(self.coeff))])
        return sum([Monome(n-1, n*self.coeff[n]) for n in range(len(self.coeff))]).derivative(k-1)
    
    def primitive(self, k=1) :
        if k==1 :
            return sum([Monome(n+1, Fraction(self.coeff[n],n+1)) if n!=0 else Monome(1,self.coeff[n]) for n in range(len(self.coeff))])
        return sum([Monome(n+1, Fraction(self.coeff[n],n+1)) for n in range(len(self.coeff))]).primitive(k-1)
    
    def display(self, xmin, xmax) :
        print(f"\n{self}")
        x = np.linspace(xmin, xmax, 300)
        ydist = max(self(x, False))-min(self(x, False)) 
        plt.ylim(min(self(x, False))-0.1*ydist, max(self(x, False))+0.1*ydist)
        plt.axvline(x=0, c="black")
        plt.axhline(y=0, c="black")
        plt.plot(x, self(x, False))
    
    def real(self) :
        l=Polynomial(self.coeff[::-1])
        for n in range(len(l.coeff)):
            if isinstance(l.coeff[n], Complex):
                l.coeff[n]=l.coeff[n].real
        return l
    
    def imag(self) :
        l=Polynomial(self.coeff[::-1])
        for n in range(len(l.coeff)):
            if isinstance(l.coeff[n], Complex):
                l.coeff[n]=l.coeff[n].imag
        return l

def Monome(degree, coeff):
    """
    Permet d'initialiser plus simplement un monôme de degré et de coefficient choisi :
    Renvoit un objet de la classe Polynomial. 
    """
    return Polynomial([coeff]+[0]*degree)

def racines(*racines) :
    """
    Permet de définir un polynome par ses racines
    """
    if isinstance(racines[0], list) :
        racines = racines[0]
    return reduce(mul, [Polynomial(1,-r) for r in racines])

###CONSTANTE ET BEZIER LINEAIRE###

i=Complex(0,1)
x=Polynomial(1,0)

linear_bez = lambda alpha, omega : alpha*Polynomial(-1,1)+omega*Polynomial(1,0)

Complex(3,-2)/Complex(2,5)


#########  Mises en application  ##########

def trianglepascal(n) :
    """
    Une simple formalité, on imprime la liste de coefficients des (x+1)^k, pour k variant de 0 à n
    """
    print([1])
    for k in range(1, n+1) :
        print((Polynomial(1,1)**k).coeff)

def interpolation(*points, display=False, var="x") :
    """
    Méthode d'interpolation polynomiale Lagrangienne d'une liste de n instances de la classe Points
    Affichera la représentation graphique d'un polynome de degré n-1 au plus, passant par tous les points.
    """
    if isinstance(points[0], list) :
        points = points[0]
    if isinstance(points[0], Complex) :
        points = [p.image() for p in points]
    assert all(isinstance(pt, Point) for pt in points)
    points = list(points)
    functs = []
    for excl in points :
        f = racines([pt.x for pt in [n for n in points if n!=excl]])
        try :
            f = f*excl.y/(f(excl.x))
        except ZeroDivisionError :
            print("Deux de vos points sont de même abscisses ! Vous ruinez tout !!")
            return None
        functs.append(f)
    total = sum(functs)
    if display :
        total.display(min([pt.x for pt in points]),max([pt.x for pt in points]))
        plt.scatter([float(pt.x) for pt in points], [float(pt.y) for pt in points])
        for pt in points :
            plt.annotate(f"({pt.x}, {pt.y})",(pt.x, pt.y),(pt.x, pt.y), 'data', 'offset points')
        plt.title(f"f(x)={total}", fontsize=10)
        plt.fullscreen()
        plt.show()
    return Polynomial(total.coeff[::-1], var = var)

def bezcurve(*points) :
    if isinstance(points[0], list) or isinstance(points[0], tuple) :
        points = list(points[0])
    assert all([isinstance(n, Point) for n in points]) or all([isinstance(n,Complex) for n in points]) or all([isinstance(n,Polynomial) for n in points])
    if isinstance(points[0], Point) :
        afx = [pt.affixe() for pt in points]
    if len(points)==2 :
        return linear_bez(points[0],points[1])
    return bezcurve([linear_bez(points[n], points[n+1]) for n in range(len(points)-1)])

def showbezier(*points, showpoints=False):
    if isinstance(points[0], list) or isinstance(points[0], tuple) :
        points = list(points[0])
    m=bezcurve(points)
    t=np.linspace(0,1,300) #Definition de la variable t, de 0 à 1
    plt.plot(m.real()(t,False), m.imag()(t,False))
    if showpoints :
        if isinstance(points[0], Point) :
            points = [pt.affixe() for pt in points]
        plt.scatter([float(pt.real) for pt in points], [float(pt.imag) for pt in points], color = "red")
        for n in range (len(points)) :
            plt.annotate(f"P{n}({points[n].real}, {points[n].imag})",(points[n].real, points[n].imag),(points[n].real, points[n].imag), 'data', 'offset points')
    plt.show()
    

def sum_k_pow(n):
    l = [0]
    for k in range(1, n+2) : #on fait la somme de k^n. Cela fait un polynôme de degré n+1.
        l.append(l[-1]+k**n) #D'après Lagrange, il faut donc n+2 conditions à imposer. On a déjà une C.I, reste n+1 pts.
    pts = [k+i*l[k] for k in range(len(l))]
    print(pts)
    return interpolation(pts, var="n")