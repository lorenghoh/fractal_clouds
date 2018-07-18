import ujson as json

with open('model_config.json', 'r') as file:
    try:
        config = json.load(file)
    except:
        raise("JSON config not found")
    
    c = config['config']
    del config['config']

if __name__ == '__main__':
    print(c)
    print(config)