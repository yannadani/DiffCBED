import numpy as np
from graphical_models.classes.interventions.interventions import ConstantIntervention

class D:
    def __init__(self, func, *args, **keywords):
        def newfunc(*fargs, **fkeywords):
            newkeywords = {**keywords, **fkeywords}
            return func(*args, *fargs, **newkeywords)

        newfunc.func = func
        newfunc.args = args
        newfunc.keywords = keywords
        self.partial = newfunc

    def sample(self, size):
        return self.partial(size=size)


class Constant(ConstantIntervention):
    def __init__(self, value):
        self.val = value
        self.value = value