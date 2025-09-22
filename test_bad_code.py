# Intentionally bad code to test quality enforcement
import os,sys   # Bad import style
import hashlib

def badFunction( x,y ):   # Bad function naming and spacing
    password = "hardcoded_password123"    # Security issue
    print("Debug statement")   # Print statement
    if x==y:  # Missing spaces
        return x+y   # Missing spaces
    else:
        exec("print('security issue')")   # Security issue
        return None

# Missing docstring
class BadClass:
    def __init__(self):
        pass

# Long line that exceeds 88 characters and should be flagged by the linter for line length violations
x = 1