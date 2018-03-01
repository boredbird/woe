Dataset Information
===================

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

**YOU SHOULD SPECIFY THE VARIABLES DTYPES WITH config.csv**

Appointment:

continuous variables: is_tobe_bin=1 and is_candidate=1

discrete variables: is_tobe_bin=0 and is_candidate=1

Content
=======

There are 25 variables:

 * ID: ID of each client
 * LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
 * SEX: Gender (1=male, 2=female)
 * EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
 * MARRIAGE: Marital status (1=married, 2=single, 3=others)
 * AGE: Age in years
 * PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
 * PAY_2: Repayment status in August, 2005 (scale same as above)
 * PAY_3: Repayment status in July, 2005 (scale same as above)
 * PAY_4: Repayment status in June, 2005 (scale same as above)
 * PAY_5: Repayment status in May, 2005 (scale same as above)
 * PAY_6: Repayment status in April, 2005 (scale same as above)
 * BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
 * BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
 * BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
 * BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
 * BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
 * BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
 * PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
 * PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
 * PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
 * PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
 * PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
 * PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
 * default.payment.next.month: Default payment (1=yes, 0=no)
