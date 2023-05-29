
class Design:
    def __init__(self, node, value):
        self.node = node
        self.value = value

    def __iter__(self):
        yield self.node
        yield self.value


class Designs:
    def __init__(self):
        self.designs = []

    def add(self, node, value):
        self.designs.append(Design(node, value))

    def __iter__(self):
        return iter(self.designs)
