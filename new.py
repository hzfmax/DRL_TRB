from utils.loader import get_victoria_data

data1 = get_victoria_data()
data = data1
name = data['name']
length = data['length']
routes = data['routes']
run = data['run']
distance = data['distance']
demand = data['demand']

name = 'A'
data['name'] = name
print(data["name"])

# print((length))
# length = 5
# print((length))
# data['length'] = length
# print(data["length"])

# print((routes))
# print(len(routes))
# del routes[0:3]
# del routes[0+8:3+8]
# print((routes))
# print(len(routes))
# data['routes'] = routes
# print(data["routes"])

# run[3] = 2000
# data['run'] = run
# print(data["run"])
