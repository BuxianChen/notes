# C++primerplus\(5ed\)



```cpp
int divide(int dividend, int divisor) {
    if (dividend == INT_MIN && divisor == -1) {
        return INT_MAX;
    }
    long dvd = labs(dividend), dvs = labs(divisor), ans = 0;
    int sign = dividend > 0 ^ divisor > 0 ? -1 : 1;
    while (dvd >= dvs) {
        long temp = dvs, m = 1;
        while (temp << 1 <= dvd) {
            temp <<= 1;
            m <<= 1;
        }
        dvd -= temp;
        ans += m;
    }
    return sign * ans;
}


int divide(int A, int B) {
    if (A == INT_MIN && B == -1) return INT_MAX;
    int a = abs(A), b = abs(B), res = 0, x = 0;
    while (a - b >= 0) {
        for (x = 0; a - (b << x << 1) >= 0; x++);
        res += 1 << x;
        a -= b << x;
    }
    return (A > 0) == (B > 0) ? res : -res;
}
```

