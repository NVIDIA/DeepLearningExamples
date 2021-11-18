try:
    profile
except NameError:
    def profile(func):
        return func
