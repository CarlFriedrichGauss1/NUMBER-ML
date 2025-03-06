not_mandatory = [9, 9, 10, 10, 7]
mandatory = [8, 9, 6, 6, 5, 9, 9] 

sum_of_weights = 0
sum_of_mandatory = 0
sum_of_not_mandatory = 0

for i in mandatory:
    sum_of_mandatory += 2 * i  
    sum_of_weights += 2      

for i in not_mandatory:
    sum_of_not_mandatory += 1.5 * i  
    sum_of_weights += 1.5            

gpa = (sum_of_not_mandatory + sum_of_mandatory) / sum_of_weights
print(gpa)

gpa_without = (sum(mandatory) + sum(not_mandatory))/(len(mandatory) + len(not_mandatory))
print(gpa_without)
