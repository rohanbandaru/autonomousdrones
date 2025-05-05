import math
import numpy as np


class Complex:
    """
    A lightweight complex‑number class with both rectangular and polar support.

    Constructors
    ------------
    Complex(a, b)              -> a + b i          (rectangular)
    Complex(r, theta, polar=True) -> r·e^{iθ}    (polar)
    Complex(theta)             -> e^{iθ}          (unit‑circle)

    Use Complex.from_polar(r, theta) for an explicit polar constructor.
    """

    # ---------- construction ----------
    def __init__(self, *args, polar: bool = False):
        if polar:                              # polar form
            if len(args) != 2:
                raise ValueError("Polar form needs (r, θ)")
            r, theta = args
            self.re = r * math.cos(theta)
            self.im = r * math.sin(theta)
        else:                                  # rectangular or unit‑circle
            if len(args) == 1:                 # unit‑circle e^{iθ}
                theta, = args
                self.re = math.cos(theta)
                self.im = math.sin(theta)
            elif len(args) == 2:               # a + b i
                self.re, self.im = args
            else:
                raise ValueError("Rectangular form needs (a, b) or single θ")

    # ---------- convenience makers ----------
    @classmethod
    def from_polar(cls, r: float, theta: float) -> "Complex":
        """Explicit polar constructor."""
        return cls(r, theta, polar=True)

    # ---------- basic properties ----------
    def magnitude(self) -> float:
        return math.hypot(self.re, self.im)

    # alias required by the prompt
    modulus = magnitude

    def argument(self) -> float:
        return math.atan2(self.im, self.re)

    def normalize(self) -> "Complex":
        m = self.magnitude()
        if m == 0:
            raise ZeroDivisionError("Cannot normalize the zero vector")
        return Complex(self.re / m, self.im / m)

    # ---------- arithmetic helpers ----------
    def add(self, other: "Complex | float | int") -> "Complex":
        other = other if isinstance(other, Complex) else Complex(other, 0)
        return Complex(self.re + other.re, self.im + other.im)

    def subtract(self, other: "Complex | float | int") -> "Complex":
        other = other if isinstance(other, Complex) else Complex(other, 0)
        return Complex(self.re - other.re, self.im - other.im)

    def multiply(self, other: "Complex | float | int") -> "Complex":
        other = other if isinstance(other, Complex) else Complex(other, 0)
        return Complex(self.re * other.re - self.im * other.im,
                       self.re * other.im + self.im * other.re)

    def inverse(self) -> "Complex":
        mag_sq = self.magnitude() ** 2
        if mag_sq == 0:
            raise ZeroDivisionError("Zero has no multiplicative inverse")
        return Complex(self.re / mag_sq, -self.im / mag_sq)

    def divide(self, other: "Complex | float | int") -> "Complex":
        return self.multiply(other.inverse() if isinstance(other, Complex) else Complex(other, 0).inverse())

    def conjugate(self) -> "Complex":
        return Complex(self.re, -self.im)

    def power(self, n: int) -> "Complex":
        r = self.magnitude() ** n
        theta = self.argument() * n
        return Complex.from_polar(r, theta)
    

    # ---------- dunder sugar ----------
    __abs__   = magnitude
    __add__   = add
    __sub__   = subtract
    __mul__   = multiply
    __truediv__ = divide
    __pow__   = power

    def __eq__(self, other):
        other = other if isinstance(other, Complex) else Complex(other, 0)
        return math.isclose(self.re, other.re) and math.isclose(self.im, other.im)

    # readable REPL / print‑outs
    def __repr__(self):
        return f"Complex({self.re:+g} {self.im:+g}i)"


if __name__ == "__main__":
    z1 = Complex(3, 4)                     # 3 + 4i
    z2 = Complex.from_polar(2, math.pi/4)  # 2·e^{iπ/4}
    print(z1.magnitude())                  # 5.0
    print(z1.normalize())                  # ≈ 0.6 + 0.8i
    print(z1.add(z2))                      # vector addition
    print(z1 * z2)                         # operator sugar for multiply
    print(z1.inverse())                    # multiplicative inverse
    print(Complex(math.pi/2))              # e^{iπ/2}  => ≈ 0 + 1i
