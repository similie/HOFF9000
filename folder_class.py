import os

class folder ():     
    def create (self): 
        os.makedirs(self, mode=0o777, exist_ok=True)
        parent = os.getcwd()
        path = os.path.join(parent, self)

        return path
