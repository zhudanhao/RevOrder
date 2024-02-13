import random
import operator
import json
import re

def complete_division_process(formula):
    """
    Given a division formula like "1235/5=247", complete the division process.
    
    :param formula: A string representing the division, e.g., "1235/5=247".
    :return: A string representing the complete process of division.
    """
    # Split the formula into dividend, divisor, and quotient
    parts = formula.split('=')
    division = parts[0].split('/')
    dividend, divisor = int(division[0]), int(division[1])
    quotient = int(parts[1])

    # Initialize the process string
    process_str = f"{dividend}/{divisor}="
    current_remainder = dividend
    cnt=1
    # Iterate over each digit in the quotient
    for i, digit in enumerate(str(quotient)):
        # Calculate the power of 10 for the current digit
        power_of_ten = 10 ** (len(str(quotient)) - i - 1)
        #print(power_of_ten)
        # Calculate the partial product
        partial_product = int(digit) * divisor * power_of_ten
        # Calculate the new remainder
        new_remainder = current_remainder - partial_product
        # Add the current step to the process string
        if cnt == 1:
            if int(digit) < 9:
                partial_product_2 = (int(digit)+1) * divisor * power_of_ten
                process_str += f"{int(digit)}sel([r{reverse_num(partial_product)}][r{reverse_num(partial_product_2)}])\n"
            else:
                process_str += f"{int(digit)}sel([r{reverse_num(partial_product)}][r{reverse_num(partial_product)}])\n"
            process_str += f"{int(digit)}rem({current_remainder}-{int(digit)}*{divisor*power_of_ten}[r{reverse_num(partial_product)}][r{reverse_num(new_remainder)}])\n"
            cnt = 2
        else:
            process_str += f"{int(digit)}rem({current_remainder}-{int(digit)}*{divisor*power_of_ten}[r{reverse_num(partial_product)}][r{reverse_num(new_remainder)}])\n"
        # Update the current remainder for the next iteration
        current_remainder = new_remainder

    # Remove the last "=" and add the final quotient
    process_str = process_str.rstrip('=') + f"={quotient}"

    return process_str

def reverse_num(a):
    if type(a) == str:
        a = int(a)
    '''
    if a<0:
        return '-'+str(-a)[::-1]
    else:
        return str(a)[::-1]
    '''
    return str(a)[::-1]
    

def reverse_result(eq):
    left = eq.split('=')[0]+'='
    result = eq.split('=')[1]
    return left + 'r'+reverse_num(result) + '=' + result

def expand_add(left,right):
    return left+'='+'r'+reverse_num(right)+'='+right

def expand_sub(left,right):
    #return left+'='+'r'+reverse_num(right)+'='+right
    num = reverse_num(right)
    
    return left+'='+'r'+num+'='+right

'''
def expand_mul_equation(left):
    num1,num2 = left.split('*')
    num1,num2 = int(num1),int(num2)
    num2_str = str(num2)

   
    results = []
    for i, digit in enumerate(num2_str):
        
        result = num1 * int(digit) * (10 ** (len(num2_str)-1-i))
        results.append(result)

    
    equation = f"{num1}*{num2}=" + "+".join(f"*{int(digit)*(10**(len(num2_str)-1-i))}" for i, digit in enumerate(num2_str)) + \
               "=" + "+".join(str(result) for result in results) + \
              "=" + str(sum(results[:2])) + "+" + str(sum(results[2:])) + \
               "=" + str(sum(results))

    return equation
'''
def expand_mul_equation(left):
    num1,num2 = left.split('*')
    a,b = int(num1),int(num2)
    
    b_digits = [int(digit) for digit in str(b)]

    
    partial_products = [a * (digit * (10 ** (len(b_digits) - i - 1))) for i, digit in enumerate(b_digits) if digit != 0]

    
    expression = f"{a}*{b}=" + "+".join([f"{a}*{digit * (10 ** (len(b_digits) - i - 1))}" for i, digit in enumerate(b_digits) if digit != 0])

    
    expression += "=" + "+".join(map(str, partial_products))

   
    while len(partial_products) > 1:
        new_partial_products = []
        for i in range(0, len(partial_products), 2):
            if i + 1 < len(partial_products):
                new_partial_products.append(partial_products[i] + partial_products[i + 1])
            else:
                new_partial_products.append(partial_products[i])
        partial_products = new_partial_products
        expression += "=" + "+".join(map(str, partial_products))

    return expression



def reverse_and_prefix(input_string):
    # Find the position of the second equals sign
    equals_indices = [i for i, x in enumerate(input_string) if x == '=']
    second_equals_index = equals_indices[1] if len(equals_indices) > 1 else None

    # Process the part after the second equals sign
    if second_equals_index is not None:
        # Split the part into segments by non-digits
        segments = re.split('(\D+)', input_string[second_equals_index + 1:])

        # Process each segment
        for i in range(len(segments)):
            if segments[i].isdigit():
                # Reverse the digits and add the 'r' prefix
                segments[i] = 'r' + segments[i][::-1]

        # Replace the part in the original string
        input_string = input_string[:second_equals_index + 1] + ''.join(segments)

    return input_string

def expand_mul(left,right):
    eq = expand_mul_equation(left)
    eq = reverse_and_prefix(eq)
    eq = eq + '=' + right
    return eq

def generate_div(n_digits, num_eq,digits2=None):
    equations = []

    for _ in range(num_eq):
        
        if digits2 == None:
            b_digits = random.randint(1, n_digits - 1)
        else:
            b_digits = digits2

        
        c_digits = n_digits - b_digits

        
        b = random.randint(10**(b_digits - 1), 10**b_digits - 1)
        c = random.randint(10**(c_digits - 1), 10**c_digits - 1)

       
        a = b * c

        equations.append(f"{a}/{b}={c}")

    return equations

def complete_division_process(formula):
        """
        Given a division formula like "1235/5=247", complete the division process.

        :param formula: A string representing the division, e.g., "1235/5=247".
        :return: A string representing the complete process of division.
        """
        # Split the formula into dividend, divisor, and quotient
        parts = formula.split('=')
        division = parts[0].split('/')
        dividend, divisor = int(division[0]), int(division[1])
        quotient = int(parts[1])

        # Initialize the process string
        process_str = f"{dividend}/{divisor}="
        current_remainder = dividend

        # Iterate over each digit in the quotient
        for i, digit in enumerate(str(quotient)):
            # Calculate the power of 10 for the current digit
            power_of_ten = 10 ** (len(str(quotient)) - i - 1)
            # Calculate the partial product
            partial_product = int(digit) * divisor * power_of_ten
            # Calculate the new remainder
            new_remainder = current_remainder - partial_product
            # Add the current step to the process string
            process_str += f"{int(digit)}R(-{int(digit)}*{divisor*power_of_ten}~r{reverse_num(partial_product)}~r{reverse_num(new_remainder)})#"
            # Update the current remainder for the next iteration
            current_remainder = new_remainder

        # Remove the last "=" and add the final quotient
        process_str = process_str[:-1]
        process_str = process_str.rstrip('=') + f"={quotient}"

        return process_str

def add_an_error(s):
        steps = s.split('=')[1].split('#')

        #random select one step to corrupt
        corrupt_step_id = random.randint(0,len(steps)-1)    
        step = steps[corrupt_step_id]
        corrupt_number = int(step[0])

        if corrupt_number == 0:
            target_number = 1
        elif corrupt_number == 9:
            target_number = 8
        else:
            target_number = corrupt_number + random.choice([-1,1])

        #print(step,target_number)

        if corrupt_step_id == 0:
            last_value = int(s.split('/')[0])
        else:

            last_value = int(steps[corrupt_step_id-1].split('~')[-1][1:-1][::-1])

        plus_num =  int(step.split('*')[1].split('~')[0])


        error_str = f'{target_number}R(-{target_number}*{plus_num}~r{reverse_num(target_number*plus_num)}~r{reverse_num(last_value-target_number*plus_num)})W'

        steps.insert(corrupt_step_id,error_str)
        return  s.split('=')[0] + '=' + '#'.join(steps) + '='+s.split('=')[-1]
    
def expand_div(left,right):
    return complete_division_process(left+'='+right)

def expand_eqs(eq_list,error_rate=0.5):
    #transform the eqs with revorder
    #error_rate: if division, has a rate of error_rate to generate a error-and-corrected formular
    trans_eq_list = []
    for eq in eq_list:
        left,right = eq.split('=')
        if '+' in left:
            trans_eq_list.append(expand_add(left,right))
        if '-' in left:
            trans_eq_list.append(expand_sub(left,right))            
        if '*' in left:
            trans_eq_list.append(expand_mul(left,right))
        if '/' in left:
            f = expand_div(left,right)            
            trans_eq_list.append(f)
            if random.uniform(0,1)<error_rate:
                e_f = add_an_error(f)
                trans_eq_list.append(e_f)
    return trans_eq_list
            
def generate_eqs(etype='+', num=100, digits=5,digits2=None):
    equations = []
    if etype == '/':
        return generate_div(digits,num,digits2=digits2)
    
    for _ in range(num):
        a = random.randint(1, 10**digits - 1)
        
        if digits2 is None:
            b = random.randint(1, 10**digits - 1)
        else:
            b = random.randint(1, 10**digits2 - 1)

        if etype == '+':
            result = a + b
            equation = f"{a}+{b}={result}"
        elif etype == '-':
            #if b>a:
            #    a,b = b,a
            
            result = a - b
            equation = f"{a}-{b}={result}"
        elif etype == '*':
            result = a * b
            equation = f"{a}*{b}={result}"
        else:
            raise ValueError("Unsupported operation type. Choose '+', '-', '*', or '/'.")

        equations.append(equation)

    return equations



                
                
class build_equations():
    def __init__(self):
        self.raw_list = [] ## for each type of raw equations.
        
        self.raw_train_list = []
        
        self.train_data = []
        self.test_data = {}
    
    def create_eqs(self,etype,num_total_per_etype,num_test_per_etype=500,digits=5,digits2=None):
        #random generate all equations for train and test
        all_eqs = generate_eqs(etype=etype, num=num_total_per_etype, digits=digits,digits2=digits2)
        random.shuffle(all_eqs)
        if num_test_per_etype>0:
            if digits2 is None:
                name = etype+ str(digits)
            else:
                name = etype+ str(digits) + '_' + str(digits2)
            self.test_data[name] = all_eqs[-num_test_per_etype:]
            self.raw_train_list.extend(all_eqs[:-num_test_per_etype])
        else:
            self.raw_train_list.extend(all_eqs)
        

        

    def build_dataset(self):
        self.train_data = expand_eqs(self.raw_train_list)
        random.shuffle(self.train_data)
        
                    
        
        
    def write_files(self,train_file,test_file,num_of_eqs_per_sample=100,epoches=3):
        train_samples  =  self.train_eqs * epoches
        random.shuffle(train_samples)
        
        train_lines = []
        line = ''
        i  =  0
        for train_sample in train_samples:
            if i < num_of_eqs_per_sample:
                line = line  + train_sample + ','
                i += 1
            else:
                if line[-1] == ',':
                    line = line[:-1]
                train_lines.append(line)
                line = train_sample + ','
                i=1
        
        line = line[:-1]
        train_lines.append(line)
        with open(train_file, 'w') as f:
            json.dump(train_lines,f)
        
        test_lines = [sample.split('=')[0]+'=' for sample in self.test_eqs]
        with open(test_file, 'w') as f:
            json.dump(test_lines,f)



