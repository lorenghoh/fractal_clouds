import ujson as json

class attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

with open('model_config.json', 'r') as file:
    try:
        config = attrdict(json.load(file))
    except:
        raise("JSON config not found")
    
    c = attrdict(config['config'])
    del config['config']

if __name__ == '__main__':
    print(c)
    print(config)