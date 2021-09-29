import math


def calculate_Perimeter(profundidad, ancho):
    a, b = profundidad, ancho

    perimeter = math.pi * ( 3*(a + b) - math.sqrt( (3*a + b) * (a + 3*b) ))
    return perimeter