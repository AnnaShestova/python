#Algorithm to get days between 2 dates

def isLeapYear(year):
     """  cc"""
     if year % 400 == 0:
         return True
     if year % 100 == 0:
         return False
     if year % 4 == 0:
         return True
     return False
 
def daysInMonth(year, month):
     if month == 1 or month == 3 or month == 5 or month == 7  or month == 8 or month == 10 or month == 12:
         return 31
     else:
         if month == 2:
             if isLeapYear(year) == True:
                 return 29
             return 28
     return 30

def nextDay(year, month, day):
     """ cc """
     if day < daysInMonth(year, month):
        return year, month, day + 1
        if month < 12:
            return year, month + 1, 1
        return year + 1, 1, 1
    
def dateIsBefore(year1, month1, day1, year2, month2, day2):
    """Returns True if year1-month1-day1 is before
       year2-month2-day2. Otherwise, returns False."""
    if year1 < year2:
        return True
    if year1 == year2:
        if month1 < month2:
            return True
        if month1 == month2:
            return day1 < day2
    return False  
    
def daysBetweenDates(year1, month1, day1, year2, month2, day2):
    """ Return the number of days between year1-month1-day1 
    and year2-month2-day2. Assume inputs are valid dayes in 
    Georgian calendar, and the first date is not after the second."""       
    days = 0
    while dateIsBefore(year1, month1, day1, year2, month2, day2):
        year1, month1, day1 == nextDay(year1, month1, day1)
        days += 1
    return days
        
def test():
    
    test_cases = [((2012,9,30,2012,10,30),30), 
                  ((2012,1,1,2013,1,1),360),
                  ((2012,9,1,2012,9,4),3),
                  ((2013,1,1,1999,12,31), "AssertionError")]
    
    for (args, answer) in test_cases:
        try:
            result = daysBetweenDates(*args)
            if result == answer and answer != "AssertionError":
                print("Test case passed!")
            else:
                print("Test with data:", args, "failed")
    
        except AssertionError:
            if answer == "AssertionError":
                print("Nice job! Test case {0} correctly raises AssertionError!\n".format(args))
            else:
                print("Check your work! Test case {0} should not raise AssertionError!\n".format(args))   
                
test()