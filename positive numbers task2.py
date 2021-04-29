list1 = [12, -7, 5, 64, -14]
new_list1 = list(filter(lambda x: (x >= 0), list1)) 
print(*new_list1) 
list2 = [12, 14, -95, 3]
new_list2 = list(filter(lambda x: x >0, list2))
print(new_list2)
	
