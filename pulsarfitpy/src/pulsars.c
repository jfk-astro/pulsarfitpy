#include "pulsars.h"
#include <stdio.h>

double add_numbers(double a, double b) {
    return a + b;
}

double multiply_numbers(double a, double b) {
    return a * b;
}

double divide_numbers(double a, double b) {
    if(b != 0) {
        return a / b;
    }

    return 0;
}