# Python-Math-Modules
Python Modules of my own creation. Particularly useful for formal calculations.

***********************************************************************************

This series of modules includes :
- A class that implements fractions into python. To use a fraction a/b in a formula,
  use Fraction(a,b). They can obviously be multiplied, divided, added, raised to any
  integer power.
- A class that implements Complex numbers. To use a Complex, a+ib, you have the choice
  either to use the syntax Complex(a,b) or a+i*b (where i is the instance Complex(0,1))
  Note that by default, any real number x is considered as the complex Complex(x,0)
- A class that implements Points in a plane 2D surface. This implementation is analogous
  to the implementation of Complex numbers, an many methods for Points use Complex.
- A class that implements Polynomials. They can be multiplied, added, raised to any power.
  Coefficients can be real, fractionnal, and even complex. Rational fractions of polynomials
  are not yet implemented. You may also use composition between two Polynomials 

***********************************************************************************

Some applications are included in this program :
- The function trianglepascal generates the n first lines of Pascal's triangle by developping
  (1+x)^n and picking the coefficients of the obtained polynomial.
- The function interpolation implements Lagrange's polynomial interpolation technique.
  Given a list of n instances of either the Complex or Fraction class, it shows the
  graph and the equation of the polynomial of degree n-1 or under which interpolates
  the given list of points.
- The function bezcurve builds the parametric complex equation of the Bezier curve formed
  by a given list of n complex number or points.
- The function sum_k_pow gives, for n a positive integer , the expression of the sum of the
  k^n under a polynomial form.
