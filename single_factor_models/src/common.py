# simple mathematical functions
def factorial(a):
    if a == 0:
        return 1
    else:
        return a*factorial(a-1)

def binomial_factor(a,b):
    return factorial(a)/(factorial(b)*factorial(a-b))


def get_var_loss(y, var_values):
     for (loss, var_value) in var_values:
         if var_value >= y:
             return loss
