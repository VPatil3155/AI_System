import unittest

from calculator import add, subtract, multiply, divide


class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(add(2, 3), 5)
        self.assertAlmostEqual(add(2.5, 0.5), 3.0)

    def test_subtraction(self):
        self.assertEqual(subtract(5, 2), 3)
        self.assertAlmostEqual(subtract(5.5, 2.2), 3.3)

    def test_multiplication(self):
        self.assertEqual(multiply(4, 3), 12)
        self.assertAlmostEqual(multiply(2.5, 4), 10.0)

    def test_division(self):
        self.assertEqual(divide(10, 2), 5)
        self.assertAlmostEqual(divide(7, 2), 3.5)

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            divide(5, 0)


if __name__ == "__main__":
    unittest.main()
