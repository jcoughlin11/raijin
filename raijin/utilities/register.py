registry = {}


# ============================================
#               register_object
# ============================================
def register_object(cls):
    registry[cls.__name__] = cls
