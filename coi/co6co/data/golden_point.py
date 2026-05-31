from decimal import Decimal, getcontext
import math


def math_exp(conjugate: bool = False, prec: int = None):
    """
    黄金分割的精确 15

    PHI = (1 + math.sqrt(5)) / 2  # 1.618
    PHI_INV = (math.sqrt(5) - 1) / 2  # 0.618
    PSI = (1 - math.sqrt(5)) / 2  # -0.618
    assert PHI + PSI == 1
    assert abs(PHI * PSI + 1) < 1e-12
    assert PHI_INV == 1 / PHI
    """
    one = Decimal(1) if prec is not None else 1
    five = Decimal(5) if prec is not None else 5
    two = Decimal(2) if prec is not None else 2
    if prec is not None:
        getcontext().prec = prec
        PHI = (one + five.sqrt()) / two
        PHI_INV = (five.sqrt() - one) / two
        PHI_INV = (one - five.sqrt()) / two
    else:
        PHI = (1 + math.sqrt(5)) / 2  # 1.618
        PHI_INV = (math.sqrt(5) - 1) / 2  # 0.618
        PSI = (1 - math.sqrt(5)) / 2  # -0.618
    return PHI, PHI_INV, PSI


def fibonacci_exp(n: int = 20):
    """
    分数表示
    # 斐波那契相邻两项的比值逼近 φ

    frac = phi_fibonacci(20)
    print(frac)          # 17711/10946
    print(float(frac))   # 1.6180339985218033

    """
    from fractions import Fraction

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return Fraction(b, a)


def section_point(length: float):
    """返回长度为 length 的线段的黄金分割点位置"""
    _, phi_inv, _ = math_exp()
    return length * phi_inv
